import torch
import math
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.datasets import MultimodalPepTCRDataset
from src.multimodal_models import BSSVAE
from src.torch_utils import save_checkpoint, load_checkpoint, save_model_full, load_model_full, mask_modality, \
    filter_modality, batch_generator, paired_batch_generator
from src.utils import epoch_counter, get_loss_metric_text
from src.metrics import reconstruction_accuracy, get_metrics, model_reconstruction_stats, BSSVAELoss, \
    reconstruct_and_compute_accuracy
from torch import cuda
from torch.nn import functional as F


def eval_trimodal_step(model, criterion, valid_loader):
    model.eval()
    acum_total_loss, acum_total_recon_loss, acum_joint_kld, acum_marginal_kld, acum_triplet = 0, 0, 0, 0, 0
    alpha_reconstructed, beta_reconstructed, pep_reconstructed = [], [], []
    alpha_true, beta_true, pep_true = [], [], []
    masks_alpha, masks_beta, masks_pep = [], [], []
    # Batch should be x_alpha, x_beta, x_pep, labels (triplet), +/- pep_weights
    with torch.no_grad():
        for batch in valid_loader:
            if valid_loader.dataset.pep_weighted:
                x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels, pep_weights = batch
            else:
                x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels = batch
                pep_weights = torch.ones([len(labels)])
            x_alpha, x_beta, x_pep, labels, pep_weights = x_alpha.to(model.device), x_beta.to(model.device), x_pep.to(
                model.device), labels.to(model.device), pep_weights.to(model.device)
            mask_alpha, mask_beta, mask_pep = mask_alpha.to(model.device), mask_beta.to(model.device), mask_pep.to(
                model.device)
            x_hat_alpha, x_hat_beta, x_hat_pep, mu_joint, logvar_joint, mus_marginal, logvars_marginal = model(x_alpha,
                                                                                                               x_beta,
                                                                                                               x_pep)
            # Should this return all individual loss terms ? Including each marginal? Or just keep it simple with
            # total reconstruction, marginal KLD, joint KLD, triplet ?
            recon_loss, kld_joint, kld_marginal, triplet_loss = criterion(x_hat_alpha, x_alpha, x_hat_beta, x_beta,
                                                                          x_hat_pep, x_pep,
                                                                          mu_joint, logvar_joint, mus_marginal,
                                                                          logvars_marginal,
                                                                          mask_alpha, mask_beta, mask_pep, labels,
                                                                          pep_weights=pep_weights)
            loss = recon_loss + kld_marginal + kld_joint + triplet_loss
            # Accumulate loss and save for metrics, multiplying by shape (and later dividing by nbatch) for normalization
            acum_total_loss += loss.item() * x_pep.shape[0]
            acum_total_recon_loss += recon_loss.item() * x_pep.shape[0]
            acum_joint_kld += kld_joint.item() * x_pep.shape[0]
            acum_marginal_kld += kld_marginal.item() * x_pep.shape[0]
            acum_triplet += triplet_loss.item() * x_pep.shape[0]
            alpha_reconstructed.append(x_hat_alpha.detach().cpu())
            beta_reconstructed.append(x_hat_beta.detach().cpu())
            pep_reconstructed.append(x_hat_pep.detach().cpu())
            alpha_true.append(x_alpha.detach().cpu())
            beta_true.append(x_beta.detach().cpu())
            pep_true.append(x_pep.detach().cpu())
            masks_alpha.append(mask_alpha.detach().cpu())
            masks_beta.append(mask_beta.detach().cpu())
            masks_pep.append(mask_pep.detach().cpu())

    # Normalize losses
    acum_total_loss /= len(valid_loader.dataset)
    acum_total_recon_loss /= len(valid_loader.dataset)
    acum_joint_kld /= len(valid_loader.dataset)
    acum_marginal_kld /= len(valid_loader.dataset)
    acum_triplet /= len(valid_loader.dataset)

    # Cat reconstructions and compute metrics
    # Getting really convoluted with everything written out explicitely ...
    # Maybe there's a more implicit / fewer lines of code way to do this.
    alpha_reconstructed = torch.cat(alpha_reconstructed)
    beta_reconstructed = torch.cat(beta_reconstructed)
    pep_reconstructed = torch.cat(pep_reconstructed)
    alpha_true = torch.cat(alpha_true)
    beta_true = torch.cat(beta_true)
    pep_true = torch.cat(pep_true)
    masks_alpha = torch.cat(masks_alpha)
    masks_beta = torch.cat(masks_beta)
    masks_pep = torch.cat(masks_pep)
    alpha_metrics = model_reconstruction_stats(model.vae_alpha, alpha_reconstructed, alpha_true,
                                               return_per_element=False, modality_mask=masks_alpha)
    beta_metrics = model_reconstruction_stats(model.vae_beta, beta_reconstructed, beta_true, return_per_element=False,
                                              modality_mask=masks_beta)
    pep_metrics = model_reconstruction_stats(model.vae_pep, pep_reconstructed, pep_true, return_per_element=False,
                                             modality_mask=masks_pep)
    valid_metrics = {'alpha_' + k: v for k, v in alpha_metrics.items()}
    valid_metrics.update({'beta_' + k: v for k, v in beta_metrics.items()})
    valid_metrics.update({'pep_' + k: v for k, v in pep_metrics.items()})
    valid_loss = {'total': acum_total_loss, 'reconstruction': acum_total_recon_loss,
                  'kld_joint': acum_joint_kld, 'kld_marg': acum_marginal_kld, 'triplet': acum_triplet}
    return valid_loss, valid_metrics


def train_trimodal_step(model, criterion, optimizer, train_loader):
    model.train()
    # Terms present :
    # alpha/beta/pep reconstructions ; alpha/beta/pep marginal KLD ; joint KLD ; triplet loss (?)
    # acum_alpha_rec_loss, acum_beta_rec_loss, acum_pep_rec_loss, acum_triplet_loss = 0, 0, 0, 0
    # acum_alpha_kld, acum_beta_kld, acum_pep_kld, acum_joint_kld = 0, 0, 0, 0
    # acum_total_recon_loss, acum_total_kld_loss = 0, 0
    acum_total_loss, acum_total_recon_loss, acum_joint_kld, acum_marginal_kld, acum_triplet = 0, 0, 0, 0, 0
    alpha_reconstructed, beta_reconstructed, pep_reconstructed = [], [], []
    alpha_true, beta_true, pep_true = [], [], []
    masks_alpha, masks_beta, masks_pep = [], [], []
    # Batch should be x_alpha, x_beta, x_pep, labels (triplet), +/- pep_weights
    for i, batch in enumerate(train_loader):
        if train_loader.dataset.pep_weighted:
            x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels, pep_weights = batch
        else:
            x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels = batch
            pep_weights = torch.ones([len(labels)])
        x_alpha, x_beta, x_pep, labels, pep_weights = x_alpha.to(model.device), x_beta.to(model.device), x_pep.to(
            model.device), labels.to(model.device), pep_weights.to(model.device)
        mask_alpha, mask_beta, mask_pep = mask_alpha.to(model.device), mask_beta.to(model.device), mask_pep.to(
            model.device)
        x_hat_alpha, x_hat_beta, x_hat_pep, mu_joint, logvar_joint, mus_marginal, logvars_marginal = model(x_alpha,
                                                                                                           x_beta,
                                                                                                           x_pep)
        # Should this return all individual loss terms ? Including each marginal? Or just keep it simple with
        # total reconstruction, marginal KLD, joint KLD, triplet ?
        recon_loss, kld_joint, kld_marginal, triplet_loss = criterion(x_hat_alpha, x_alpha, x_hat_beta, x_beta,
                                                                      x_hat_pep, x_pep,
                                                                      mu_joint, logvar_joint, mus_marginal,
                                                                      logvars_marginal,
                                                                      mask_alpha, mask_beta, mask_pep, labels,
                                                                      pep_weights=pep_weights)
        loss = recon_loss + kld_marginal + kld_joint + triplet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('HERE', model.vae_beta.encoder[0].weight)
        if any([model.vae_beta.encoder[0].weight.isnan().any(),
                model.vae_alpha.encoder[0].weight.isnan().any(),
                model.vae_pep.encoder[0].weight.isnan().any()]):
            # Weights go from normal to suddenly nans
            print(model.vae_beta.encoder[0].weight)
            print(model.vae_alpha.encoder[0].weight)
            print(model.vae_pep.encoder[0].weight)
            print(i)
            import sys
            sys.exit(1)
        # Accumulate loss and save for metrics, multiplying by shape (and later dividing by nbatch) for normalization
        acum_total_loss += loss.item() * x_pep.shape[0]
        acum_total_recon_loss += recon_loss.item() * x_pep.shape[0]
        acum_joint_kld += kld_joint.item() * x_pep.shape[0]
        acum_marginal_kld += kld_marginal.item() * x_pep.shape[0]
        acum_triplet += triplet_loss.item() * x_pep.shape[0]
        alpha_reconstructed.append(x_hat_alpha.detach().cpu())
        beta_reconstructed.append(x_hat_beta.detach().cpu())
        pep_reconstructed.append(x_hat_pep.detach().cpu())
        alpha_true.append(x_alpha.detach().cpu())
        beta_true.append(x_beta.detach().cpu())
        pep_true.append(x_pep.detach().cpu())
        masks_alpha.append(mask_alpha.detach().cpu())
        masks_beta.append(mask_beta.detach().cpu())
        masks_pep.append(mask_pep.detach().cpu())

    # Normalize losses
    acum_total_loss /= len(train_loader.dataset)
    acum_total_recon_loss /= len(train_loader.dataset)
    acum_joint_kld /= len(train_loader.dataset)
    acum_marginal_kld /= len(train_loader.dataset)
    acum_triplet /= len(train_loader.dataset)

    # Cat reconstructions and compute metrics
    alpha_reconstructed = torch.cat(alpha_reconstructed)
    beta_reconstructed = torch.cat(beta_reconstructed)
    pep_reconstructed = torch.cat(pep_reconstructed)
    alpha_true = torch.cat(alpha_true)
    beta_true = torch.cat(beta_true)
    pep_true = torch.cat(pep_true)
    masks_alpha = torch.cat(masks_alpha)
    masks_beta = torch.cat(masks_beta)
    masks_pep = torch.cat(masks_pep)
    alpha_metrics = model_reconstruction_stats(model.vae_alpha, alpha_reconstructed, alpha_true,
                                               return_per_element=False, modality_mask=masks_alpha)
    beta_metrics = model_reconstruction_stats(model.vae_beta, beta_reconstructed, beta_true, return_per_element=False,
                                              modality_mask=masks_beta)
    pep_metrics = model_reconstruction_stats(model.vae_pep, pep_reconstructed, pep_true, return_per_element=False,
                                             modality_mask=masks_pep)
    train_metrics = {'alpha_' + k: v for k, v in alpha_metrics.items()}
    train_metrics.update({'beta_' + k: v for k, v in beta_metrics.items()})
    train_metrics.update({'pep_' + k: v for k, v in pep_metrics.items()})
    train_loss = {'total': acum_total_loss, 'reconstruction': acum_total_recon_loss,
                  'kld_joint': acum_joint_kld, 'kld_marg': acum_marginal_kld, 'triplet': acum_triplet}
    return train_loss, train_metrics


def predict_trimodal(model, dataset, dataloader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'
    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'
    model.eval()
    df = dataset.df.reset_index(drop=True).copy()
    alpha_reconstructed, beta_reconstructed, pep_reconstructed = [], [], []
    alpha_true, beta_true, pep_true = [], [], []
    masks_alpha, masks_beta, masks_pep = [], [], []
    z_latent_joint = []
    # Batch should be x_alpha, x_beta, x_pep, labels (triplet), +/- pep_weights
    with torch.no_grad():
        for batch in dataloader:
            if dataloader.dataset.pep_weighted:
                x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels, pep_weights = batch
            else:
                x_alpha, x_beta, x_pep, mask_alpha, mask_beta, mask_pep, labels = batch
                pep_weights = torch.ones([len(labels)])
            x_alpha, x_beta, x_pep, labels, pep_weights = x_alpha.to(model.device), x_beta.to(model.device), x_pep.to(
                model.device), labels.to(model.device), pep_weights.to(model.device)
            mask_alpha, mask_beta, mask_pep = mask_alpha.to(model.device), mask_beta.to(model.device), mask_pep.to(
                model.device)
            x_hat_alpha, x_hat_beta, x_hat_pep, mu_joint, _, _, _ = model(x_alpha, x_beta, x_pep)

            alpha_reconstructed.append(x_hat_alpha.detach().cpu())
            beta_reconstructed.append(x_hat_beta.detach().cpu())
            pep_reconstructed.append(x_hat_pep.detach().cpu())
            alpha_true.append(x_alpha.detach().cpu())
            beta_true.append(x_beta.detach().cpu())
            pep_true.append(x_pep.detach().cpu())
            masks_alpha.append(mask_alpha.detach().cpu())
            masks_beta.append(mask_beta.detach().cpu())
            masks_pep.append(mask_pep.detach().cpu())
            z_latent_joint.append(mu_joint.detach().cpu())
    # Cat masks first to mask reconstructed
    masks_alpha = torch.cat(masks_alpha)
    masks_beta = torch.cat(masks_beta)
    masks_pep = torch.cat(masks_pep)

    # Then compute each metrics by modality, then reconstruct and save
    alpha_reconstructed = mask_modality(torch.cat(alpha_reconstructed), masks_alpha, fill_value=dataset.pad_scale)
    alpha_true = mask_modality(torch.cat(alpha_true), masks_alpha, fill_value=dataset.pad_scale)
    metrics = model_reconstruction_stats(model.vae_alpha, alpha_reconstructed, alpha_true,
                                         return_per_element=True,
                                         modality_mask=masks_alpha)
    alpha_reconstructed, alpha_true = model.reconstruct_sequence(alpha_reconstructed,
                                                                 'alpha'), model.reconstruct_sequence(alpha_true,
                                                                                                      'alpha')
    df['alpha_acc'] = metrics['seq_accuracy']
    df['alpha_recon'] = alpha_reconstructed
    df['alpha_true'] = alpha_true
    # Beta
    beta_reconstructed = mask_modality(torch.cat(beta_reconstructed), masks_beta, fill_value=dataset.pad_scale)
    beta_true = mask_modality(torch.cat(beta_true), masks_beta, fill_value=dataset.pad_scale)
    metrics = model_reconstruction_stats(model.vae_beta, beta_reconstructed, beta_true,
                                         return_per_element=True,
                                         modality_mask=masks_beta)
    beta_reconstructed, beta_true = model.reconstruct_sequence(beta_reconstructed, 'beta'), model.reconstruct_sequence(
        beta_true, 'beta')
    df['beta_acc'] = metrics['seq_accuracy']
    df['beta_recon'] = beta_reconstructed
    df['beta_true'] = beta_true
    # Pep
    pep_reconstructed = mask_modality(torch.cat(pep_reconstructed), masks_pep, fill_value=dataset.pad_scale)
    pep_true = mask_modality(torch.cat(pep_true), masks_pep, fill_value=dataset.pad_scale)
    metrics = model_reconstruction_stats(model.vae_pep, pep_reconstructed, pep_true,
                                         return_per_element=True,
                                         modality_mask=masks_pep)
    pep_reconstructed, pep_true = model.reconstruct_sequence(pep_reconstructed, 'pep'), model.reconstruct_sequence(
        pep_true, 'pep')
    df['pep_acc'] = metrics['seq_accuracy']
    df['pep_recon'] = pep_reconstructed
    df['pep_true'] = pep_true

    # Save the joint latents
    z_latent_joint = torch.cat(z_latent_joint)
    df[[f'z_{i}' for i in range(z_latent_joint.shape[1])]] = z_latent_joint

    return df


def trimodal_train_eval_loops(n_epochs, tolerance, model, criterion, optimizer, train_loader, valid_loader,
                              checkpoint_filename, outdir):
    print(f'Staring {n_epochs} training cycles')
    save_checkpoint(model, checkpoint_filename, dir_path=outdir, verbose=False, best_dict=None)
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_reconstruction, best_epoch, best_val_auc, best_agg_metric = 1000, 0., 1, 0.5, 0.5
    # "best_val_losses" is a dictionary of all the various split losses
    best_val_losses, best_val_metrics, best_dict = {}, {}, {}

    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_trimodal_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_trimodal_step(model, criterion, valid_loader)
        epoch_counter(model, criterion)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs:
            text = get_loss_metric_text(e, train_loss, valid_loss, train_metric, valid_metric)
            tqdm.write(text)

        # alpha_seq_accuracy, beta_seq_accuracy, should give weight to each and lower weight to pep, like 3,3,1 ?
        # loss conditions should be taken on the total loss
        loss_condition = valid_loss['total'] <= best_val_loss + tolerance
        recon_condition = (3 * valid_metric['alpha_seq_accuracy'] + 3 * valid_metric['beta_seq_accuracy'] +
                           valid_metric['pep_seq_accuracy']) / 7
        recon_condition = recon_condition >= best_val_reconstruction - tolerance
        if e > 1 and loss_condition and recon_condition:
            best_epoch = e
            best_val_loss = valid_loss['total']
            # Taking a weighted mean here
            best_val_reconstruction = (3 * valid_metric['alpha_seq_accuracy'] + 3 * valid_metric['beta_seq_accuracy'] +
                                       valid_metric['pep_seq_accuracy']) / 7
            best_val_losses = valid_loss
            best_val_metrics = valid_metric

            best_dict = {'Best epoch': best_epoch, 'Best val loss': best_val_loss}
            best_dict.update(valid_loss)
            best_dict.update(valid_metric)
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)

    last_filename = 'last_epoch_' + checkpoint_filename
    save_checkpoint(model, filename=last_filename, dir_path=outdir, best_dict=best_dict)

    print(f'End of training cycles')
    print(best_dict)
    model = load_checkpoint(model, checkpoint_filename, outdir)
    model.eval()
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics


def train_multimodal_step(model: BSSVAE, criterion: BSSVAELoss, optimizer, train_loader):
    model.train()
    acum_total_loss, acum_recon_marg, acum_recon_joint, acum_kld_marg, acum_kld_joint = 0, 0, 0, 0, 0
    tcr_marg_recon, tcr_joint_recon, pep_marg_recon, pep_joint_recon = [], [], [], []
    tcr_marg_true, tcr_joint_true, pep_marg_true, pep_joint_true = [], [], [], []

    for batch in train_loader:
        # Pre-saves input prior to setting to device, so we don't have to detach+cpu
        tcr_marg_true.append(batch[0])
        tcr_joint_true.append(batch[1])
        pep_joint_true.append(batch[2])
        pep_marg_true.append(batch[3])
        # Assumes batch = [x_tcr_marg, x_tcr_joint, x_pep_joint, x_pep_marg]
        batch = [x.to(model.device) for x in batch]
        recons, mus, logvars = model(*batch)
        # Batch is `trues` ; Criterion takes list+dicts and returns dicts
        # Might be useful for debugging purposes to return dicts
        # TODO: Could return a single dict with the pre-summed values later
        recon_loss_marg, recon_loss_joint, kld_loss_marg, kld_loss_joint = criterion(batch, recons, mus, logvars)
        # Sum because those are dicts of two k:v pairs
        recon_loss_marg, recon_loss_joint = sum(recon_loss_marg.values()), sum(recon_loss_joint.values())
        kld_loss_marg, kld_loss_joint = sum(kld_loss_marg.values()), sum(kld_loss_joint.values())
        # Sum & update
        loss = recon_loss_marg + recon_loss_joint + kld_loss_marg + kld_loss_joint
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulating normalized loss
        acum_total_loss += loss.item() * len(batch)
        acum_recon_marg += recon_loss_marg.item() * len(batch)
        acum_recon_joint += recon_loss_joint.item() * len(batch)
        acum_kld_marg += kld_loss_marg.item() * len(batch)
        acum_kld_joint += kld_loss_joint.item() * len(batch)
        # Saving reconstruction to assess accuracy
        tcr_marg_recon.append(recons['tcr_marg'].detach().cpu())
        tcr_joint_recon.append(recons['tcr_joint'].detach().cpu())
        pep_joint_recon.append(recons['pep_joint'].detach().cpu())
        pep_marg_recon.append(recons['pep_marg'].detach().cpu())

    # normalizing
    acum_total_loss /= len(train_loader.dataset)
    acum_recon_marg /= len(train_loader.dataset)
    acum_recon_joint /= len(train_loader.dataset)
    acum_kld_marg /= len(train_loader.dataset)
    acum_kld_joint /= len(train_loader.dataset)

    # Cat and reconstruction stats
    tcr_marg_recon = torch.cat(tcr_marg_recon)
    tcr_joint_recon = torch.cat(tcr_joint_recon)
    pep_joint_recon = torch.cat(pep_joint_recon)
    pep_marg_recon = torch.cat(pep_marg_recon)
    tcr_marg_true = torch.cat(tcr_marg_true)
    tcr_joint_true = torch.cat(tcr_joint_true)
    pep_joint_true = torch.cat(pep_joint_true)
    pep_marg_true = torch.cat(pep_marg_true)
    # Getting metrics and losses
    tcr_marg_metrics = model_reconstruction_stats(model.tcr_decoder, tcr_marg_recon, tcr_marg_true,
                                                  return_per_element=False, modality_mask=None)
    tcr_joint_metrics = model_reconstruction_stats(model.tcr_decoder, tcr_joint_recon, tcr_joint_true,
                                                   return_per_element=False, modality_mask=None)
    pep_joint_metrics = model_reconstruction_stats(model.pep_decoder, pep_joint_recon, pep_joint_true,
                                                   return_per_element=False, modality_mask=None)
    pep_marg_metrics = model_reconstruction_stats(model.pep_decoder, pep_marg_recon, pep_marg_true,
                                                  return_per_element=False, modality_mask=None)
    train_metrics = {'tcr_marg_' + k: v for k, v in tcr_marg_metrics.items()}
    train_metrics.update({'tcr_joint_' + k: v for k, v in tcr_joint_metrics.items()})
    train_metrics.update({'pep_joint_' + k: v for k, v in pep_joint_metrics.items()})
    train_metrics.update({'pep_marg_' + k: v for k, v in pep_marg_metrics.items()})
    train_loss = {'total': acum_total_loss, 'recon_marg': acum_recon_marg, 'recon_joint': acum_recon_joint,
                  'kld_marg': acum_kld_marg, 'kld_joint': acum_kld_joint}
    return train_loss, train_metrics


def eval_multimodal_step(model: BSSVAE, criterion: BSSVAELoss, valid_loader):
    model.eval()
    acum_total_loss, acum_recon_marg, acum_recon_joint, acum_kld_marg, acum_kld_joint = 0, 0, 0, 0, 0
    tcr_marg_recon, tcr_joint_recon, pep_marg_recon, pep_joint_recon = [], [], [], []
    tcr_marg_true, tcr_joint_true, pep_marg_true, pep_joint_true = [], [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            # Pre-saves input prior to setting to device so we don't have to detach+cpu
            tcr_marg_true.append(batch[0])
            tcr_joint_true.append(batch[1])
            pep_joint_true.append(batch[2])
            pep_marg_true.append(batch[3])
            # Assumes batch = [x_tcr_marg, x_tcr_joint, x_pep_joint, x_pep_marg]
            batch = [x.to(model.device) for x in batch]
            recons, mus, logvars = model(*batch)
            # Batch is `trues` ; Criterion takes list+dicts and returns dicts
            recon_loss_marg, recon_loss_joint, kld_loss_marg, kld_loss_joint = criterion(batch, recons, mus, logvars)
            # Sum because those are dicts of two k:v pairs
            recon_loss_marg, recon_loss_joint = sum(recon_loss_marg.values()), sum(recon_loss_joint.values())
            kld_loss_marg, kld_loss_joint = sum(kld_loss_marg.values()), sum(kld_loss_joint.values())
            # Sum & update
            loss = recon_loss_marg + recon_loss_joint + kld_loss_marg + kld_loss_joint
            # Accumulating normalized loss
            acum_total_loss += loss.item() * len(batch)
            acum_recon_marg += recon_loss_marg.item() * len(batch)
            acum_recon_joint += recon_loss_joint.item() * len(batch)
            acum_kld_marg += kld_loss_marg.item() * len(batch)
            acum_kld_joint += kld_loss_joint.item() * len(batch)
            # Saving reconstruction to assess accuracy
            tcr_marg_recon.append(recons['tcr_marg'].detach().cpu())
            tcr_joint_recon.append(recons['tcr_joint'].detach().cpu())
            pep_joint_recon.append(recons['pep_joint'].detach().cpu())
            pep_marg_recon.append(recons['pep_marg'].detach().cpu())

    # normalizing
    acum_total_loss /= len(valid_loader.dataset)
    acum_recon_marg /= len(valid_loader.dataset)
    acum_recon_joint /= len(valid_loader.dataset)
    acum_kld_marg /= len(valid_loader.dataset)
    acum_kld_joint /= len(valid_loader.dataset)

    # Cat and reconstruction stats
    tcr_marg_recon = torch.cat(tcr_marg_recon)
    tcr_joint_recon = torch.cat(tcr_joint_recon)
    pep_joint_recon = torch.cat(pep_joint_recon)
    pep_marg_recon = torch.cat(pep_marg_recon)
    tcr_marg_true = torch.cat(tcr_marg_true)
    tcr_joint_true = torch.cat(tcr_joint_true)
    pep_joint_true = torch.cat(pep_joint_true)
    pep_marg_true = torch.cat(pep_marg_true)
    # Getting metrics and losses
    tcr_marg_metrics = model_reconstruction_stats(model.tcr_decoder, tcr_marg_recon, tcr_marg_true,
                                                  return_per_element=False, modality_mask=None)
    tcr_joint_metrics = model_reconstruction_stats(model.tcr_decoder, tcr_joint_recon, tcr_joint_true,
                                                   return_per_element=False, modality_mask=None)
    pep_joint_metrics = model_reconstruction_stats(model.pep_decoder, pep_joint_recon, pep_joint_true,
                                                   return_per_element=False, modality_mask=None)
    pep_marg_metrics = model_reconstruction_stats(model.pep_decoder, pep_marg_recon, pep_marg_true,
                                                  return_per_element=False, modality_mask=None)
    valid_metrics = {'tcr_marg_' + k: v for k, v in tcr_marg_metrics.items()}
    valid_metrics.update({'tcr_joint_' + k: v for k, v in tcr_joint_metrics.items()})
    valid_metrics.update({'pep_joint_' + k: v for k, v in pep_joint_metrics.items()})
    valid_metrics.update({'pep_marg_' + k: v for k, v in pep_marg_metrics.items()})
    valid_loss = {'total': acum_total_loss, 'recon_marg': acum_recon_marg, 'recon_joint': acum_recon_joint,
                  'kld_marg': acum_kld_marg, 'kld_joint': acum_kld_joint}

    for k in tcr_marg_metrics.keys():
        valid_metrics[f'mean_{k}'] = np.mean([tcr_marg_metrics[k], tcr_joint_metrics[k],
                                              pep_joint_metrics[k], pep_marg_metrics[k]])
    return valid_loss, valid_metrics


def predict_multimodal(model: BSSVAE, dataset: MultimodalPepTCRDataset, batch_size):
    """
    Here, it uses dataset and not dataloader in order to return the entire DF with the missing modalities!
    So it doesn't use dataloader because dataloader sub-samples the missing modalities based on epochs and we want the full thing
    Args:
        model:
        dataset:

    Returns:
        predictions_df
    """
    model.eval()
    z_latent = []
    x_recon_tcr_marg, x_true_tcr_marg = [], []
    x_recon_pep_marg, x_true_pep_marg = [], []
    x_recon_tcr_joint, x_true_tcr_joint = [], []
    x_recon_pep_joint, x_true_pep_joint = [], []
    # Original df, re-ordered
    tcr_df = dataset.df_tcr_only.copy()
    pep_df = dataset.df_pep_only.copy()
    paired_df = dataset.df_pep_tcr.copy()
    mlt = dataset.max_len_tcr
    mlp = dataset.max_len_pep
    with torch.no_grad():
        # marginal TCR first
        for batch in batch_generator(dataset.x_tcr_marg, batch_size):
            # Pre saves true_vector to avoid detaching and cpu later
            x_true_tcr_marg.append(batch)
            batch = batch.to(model.device)
            x_hat, z = model.forward_marginal(batch, which='tcr')
            x_recon_tcr_marg.append(x_hat.detach().cpu())
            z_latent.append(z.detach().cpu())
        # Concat, reconstruct, metrics
        seq_hat_true, seq_hat_recon, metrics = reconstruct_and_compute_accuracy(model.tcr_decoder, x_true_tcr_marg,
                                                                                x_recon_tcr_marg)
        tcr_df['seq_true'] = seq_hat_true
        tcr_df['seq_recon'] = seq_hat_recon
        tcr_df['seq_acc'] = metrics['seq_accuracy']
        # Marginal Peptide part
        for batch in batch_generator(dataset.x_pep_marg, batch_size):
            x_true_pep_marg.append(batch)
            batch = batch.to(model.device)
            x_hat, z = model.forward_marginal(batch, which='pep')
            x_recon_pep_marg.append(x_hat.detach().cpu())
            z_latent.append(z.detach().cpu())

        x_recon_pep_marg = torch.cat(x_recon_pep_marg)
        x_true_pep_marg = torch.cat(x_true_pep_marg)
        seq_hat_true, seq_hat_recon, metrics = reconstruct_and_compute_accuracy(model.pep_decoder, x_true_pep_marg,
                                                                                x_recon_pep_marg)
        pep_df['seq_true'] = seq_hat_true
        pep_df['seq_recon'] = seq_hat_recon
        pep_df['seq_acc'] = metrics['seq_accuracy']
        # Joint/Paired data part
        for batch in paired_batch_generator(dataset.x_tcr_joint, dataset.x_pep_joint, batch_size):
            x_true_tcr_joint.append(batch[0])
            x_true_pep_joint.append(batch[1])
            batch = [b.to(model.device) for b in batch]
            x_hat_tcr, x_hat_pep, z = model.forward_joint(*batch)
            x_recon_tcr_joint.append(x_hat_tcr.detach().cpu())
            x_recon_pep_joint.append(x_hat_pep.detach().cpu())
            z_latent.append(z.detach().cpu())

        seq_hat_true_tcr, seq_hat_recon_tcr, metrics_tcr = reconstruct_and_compute_accuracy(
            model.tcr_decoder,
            x_true_tcr_joint,
            x_recon_tcr_joint)
        seq_hat_true_pep, seq_hat_recon_pep, metrics_pep = reconstruct_and_compute_accuracy(
            model.pep_decoder,
            x_true_pep_joint,
            x_recon_pep_joint)
        # concatenate the strings TCR-PEP for true and recon
        seq_true = [a + b for a, b in zip(seq_hat_true_tcr, seq_hat_true_pep)]
        seq_recon = [a + b for a, b in zip(seq_hat_recon_tcr, seq_hat_recon_pep)]
        # Do a weighted mean based on sequence length

        accs = [(mlt * tcr_acc + mlt * pep_acc) / (mlt + mlp) for tcr_acc, pep_acc in
                zip(metrics_tcr['seq_accuracy'],
                    metrics_pep['seq_accuracy'])]
        paired_df['seq_true'] = seq_true
        paired_df['seq_recon'] = seq_recon
        paired_df['seq_acc'] = accs

    results_df = pd.concat([tcr_df, pep_df, paired_df])
    results_df[[f'z_{i}' for i in range(model.latent_dim)]] = torch.cat(z_latent)
    return results_df


def multimodal_train_eval_loops(n_epochs, model, criterion, optimizer, train_loader, valid_loader, checkpoint_filename,
                                outdir, tolerance=1e-4):
    print(f'Staring {n_epochs} training cycles')
    # Pre-save model at epoch 0
    save_checkpoint(model, checkpoint_filename, dir_path=outdir, verbose=False, best_dict=None)
    best_val_loss, best_val_reconstruction, best_epoch = 1000, 0, 0
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    intervals = (np.arange(0.2, 1, 0.2) * n_epochs).astype(int)

    for e in tqdm(range(1, n_epochs+1), desc='epochs', leave=False):
        train_loss, train_metric = train_multimodal_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_multimodal_step(model, criterion, valid_loader)
        epoch_counter(model, criterion, train_loader, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # Periodic prints for tracking
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs:
            text = get_loss_metric_text(e, train_loss, valid_loss, train_metric, valid_metric)
            tqdm.write(text)

        loss_condition = valid_loss['total'] <= best_val_loss + tolerance
        recon_condition = valid_metric['mean_seq_accuracy'] >= best_val_reconstruction - tolerance
        if e > 1 and loss_condition and recon_condition:
            best_epoch = e
            best_val_loss = valid_loss['total']
            best_val_reconstruction = valid_metric['mean_seq_accuracy']
            # Saving the actual dictionaries for logging purposes
            best_val_losses = valid_loss
            best_val_metrics = valid_metric

            best_dict = {'Best epoch': best_epoch, 'Best val loss': best_val_loss}
            best_dict.update(valid_loss)
            best_dict.update(valid_metric)
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)

        if e in intervals:
            fn = f'epoch_{e}_interval_' + checkpoint_filename
            savedict = {'epoch': e}
            savedict.update(valid_loss)
            savedict.update(valid_metric)
            save_checkpoint(model, filename=fn, dir_path=outdir, best_dict=savedict)

    last_filename = 'last_epoch_' + checkpoint_filename
    save_checkpoint(model, filename=last_filename, dir_path=outdir, best_dict=best_dict)

    print(f'End of training cycles')
    print(best_dict)
    model = load_checkpoint(model, checkpoint_filename, outdir)
    model.eval()
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics
