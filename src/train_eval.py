import torch
import math
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import src.datasets
from src.torch_utils import save_checkpoint, load_checkpoint, save_model_full, load_model_full
from src.metrics import reconstruction_accuracy
from torch import cuda
from torch.nn import functional as F


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-6, name='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_best_score = np.Inf
        self.delta = delta
        self.path = f'{name}.pt'

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            # This condition works for AUC ; checks that the AUC is below the best AUC +/- some delta
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Prev best score: ({self.prev_best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.prev_best_score = score


def invoke(early_stopping, loss, model, implement=False):
    if implement:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            return True
    else:
        return False


# noinspection PyUnboundLocalVariable
def train_model_step(model, criterion, optimizer, train_loader):
    """
    Args:
        model:
        criterion:
        optimizer:
        train_loader:

    Returns:

    """
    assert type(train_loader.sampler) == torch.utils.data.RandomSampler, 'TrainLoader should use RandomSampler!'
    model.train()
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss = 0, 0, 0, 0
    x_reconstructed, x_true = [], []
    for x in train_loader:
        if (criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion,
                                                                         'triplet_loss')) and train_loader.dataset.__class__.__name__ == 'TCRSpecificDataset':
            x, labels = x.pop(0).to(model.device), x.pop(-1).to(model.device)
            x_hat, mu, logvar = model(x)
            # TODO: CHECK THIS Set to eval mode for the extraction of the latent (? maybe not, need to try?)
            model.eval()
            z_batch = model.embed(x)
            model.train()
            recon_loss, kld_loss, triplet_loss = criterion(x_hat, x, mu, logvar, z_batch, labels)
            loss = recon_loss + kld_loss + triplet_loss
            acum_triplet_loss += triplet_loss.item() * x.shape[0]
        else:
            x = x.to(model.device)
            x_hat, mu, logvar = model(x)
            recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
            loss = recon_loss + kld_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x_reconstructed.append(x_hat.detach().cpu())
        x_true.append(x.detach().cpu())
        acum_total_loss += loss.item() * x.shape[0]
        acum_recon_loss += recon_loss.item() * x.shape[0]
        acum_kld_loss += kld_loss.item() * x.shape[0]

    # Normalizes to loss per batch
    acum_total_loss /= len(train_loader.dataset)
    acum_recon_loss /= len(train_loader.dataset)
    acum_kld_loss /= len(train_loader.dataset)

    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)
    # seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    # seq_true, v_true, j_true = model.reconstruct_hat(x_true)
    #
    # seq_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat,
    #                                                                pad_index=20)

    train_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}
    if criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion, 'triplet_loss'):
        acum_triplet_loss /= len(train_loader.dataset)
        train_loss['triplet'] = acum_triplet_loss

    train_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    return train_loss, train_metrics


# noinspection PyUnboundLocalVariable
def eval_model_step(model, criterion, valid_loader):
    model.eval()
    # disables gradient logging
    acum_total_loss, acum_recon_loss, acum_kld_loss, acum_triplet_loss = 0, 0, 0, 0
    x_reconstructed, x_true = [], []
    with torch.no_grad():
        # Same workaround as above
        for x in valid_loader:
            if (criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion,
                                                                             'triplet_loss')) and valid_loader.dataset.__class__.__name__ == 'TCRSpecificDataset':
                x, labels = x.pop(0).to(model.device), x.pop(-1).to(model.device)
                x_hat, mu, logvar = model(x)
                # Model is already in eval mode here
                z_batch = model.embed(x)
                recon_loss, kld_loss, triplet_loss = criterion(x_hat, x, mu, logvar, z_batch, labels)
                loss = recon_loss + kld_loss + triplet_loss
                acum_triplet_loss += triplet_loss.item() * x.shape[0]
            else:
                x = x.to(model.device)
                x_hat, mu, logvar = model(x)
                recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
                loss = recon_loss + kld_loss
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            acum_total_loss += loss.item() * x.shape[0]
            acum_recon_loss += recon_loss.item() * x.shape[0]
            acum_kld_loss += kld_loss.item() * x.shape[0]
    # Normalizes to loss per batch
    acum_total_loss /= len(valid_loader.dataset)
    acum_recon_loss /= len(valid_loader.dataset)
    acum_kld_loss /= len(valid_loader.dataset)
    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)
    # seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    # seq_true, v_true, j_true = model.reconstruct_hat(x_true)
    #
    # seq_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat,
    #                                                                pad_index=20)
    # valid_metrics = {'seq_accuracy': seq_accuracy, 'v_accuracy': v_accuracy, 'j_accuracy': j_accuracy}

    valid_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}

    # Only save and normalize the triplet loss if the criterion has that attribute
    if criterion.__class__.__name__ == 'CombinedVAELoss' or hasattr(criterion, 'triplet_loss'):
        acum_triplet_loss /= len(valid_loader.dataset)
        valid_loss['triplet'] = acum_triplet_loss
    valid_metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False)
    return valid_loss, valid_metrics


def model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False):
    # Here concat the list in case we are given a list of tensors (as done in the train/valid batching system)
    x_reconstructed = torch.cat(x_reconstructed) if type(x_reconstructed) == list else x_reconstructed
    x_true = torch.cat(x_true) if type(x_true) == list else x_true
    seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, v_true, j_true = model.reconstruct_hat(x_true)
    if not model.use_v:
        v_hat, v_true = None, None
    if not model.use_j:
        j_hat, j_true = None, None

    if all([hasattr(model, 'use_a'), hasattr(model, 'use_b'), hasattr(model, 'use_pep')]):
        metrics = {}
        mlb, mla, mlp = model.max_len_b, model.max_len_a, model.max_len_pep
        b_hat, a_hat, pep_hat = seq_hat[:, :mlb], seq_hat[:, mlb:mlb + mla], seq_hat[:, mlb + mla:]
        b_true, a_true, pep_true = seq_true[:, :mlb], seq_true[:, mlb:mlb + mla], seq_true[:, mlb + mla:]
        v_accuracy, j_accuracy = 0, 0  # Pre-instantiate to 0
        if model.use_a:
            a_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(a_true, a_hat, v_true, v_hat, j_true, j_hat,
                                                                         pad_index=20,
                                                                         return_per_element=return_per_element)
            metrics['a_accuracy'] = a_accuracy
        if model.use_b:
            b_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(b_true, b_hat, v_true, v_hat, j_true, j_hat,
                                                                         pad_index=20,
                                                                         return_per_element=return_per_element)
            metrics['b_accuracy'] = b_accuracy
        if model.use_pep:
            pep_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(pep_true, pep_hat, v_true, v_hat, j_true,
                                                                           j_hat,
                                                                           pad_index=20,
                                                                           return_per_element=return_per_element)
            metrics['pep_accuracy'] = pep_accuracy
        metrics['v_accuracy'] = v_accuracy
        metrics['j_accuracy'] = j_accuracy

    else:
        seq_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat,
                                                                       pad_index=20,
                                                                       return_per_element=return_per_element)
        metrics = {'seq_accuracy': seq_accuracy, 'v_accuracy': v_accuracy, 'j_accuracy': j_accuracy}
    return metrics


def predict_model(model, dataset: any([src.datasets.CDR3BetaDataset, src.datasets.PairedDataset]),
                  dataloader: torch.utils.data.DataLoader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'
    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'
    # model.eval()
    if hasattr(model, 'use_v') and hasattr(dataset, 'use_v'):
        assert (model.use_v == dataset.use_v) and (
                model.use_j == dataset.use_j), 'use_v/use_j don\'t match for model and dataset!'

    df = dataset.df.reset_index(drop=True).copy()
    x_reconstructed, x_true, z_latent = [], [], []
    with torch.no_grad():
        # Same workaround as above
        for x in dataloader:
            if dataloader.dataset.__class__.__name__ == 'TCRSpecificDataset':
                x, labels = x.pop(0).to(model.device), x.pop(-1).to(model.device)
                x_hat, mu, logvar = model(x)
                # Model is already in eval mode here
                z = model.embed(x)
            else:
                x = x.to(model.device)
                x_hat, _, _ = model(x)
                z = model.embed(x)
            x_reconstructed.append(x_hat.detach().cpu())
            x_true.append(x.detach().cpu())
            z_latent.append(z.detach().cpu())

    x_reconstructed = torch.cat(x_reconstructed)
    x_true = torch.cat(x_true)
    metrics = model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=True)

    # slice_x for PairedFVAE now returns a tuple for the sequence (beta, alpha, pep)
    x_seq_recon, _, _ = model.slice_x(x_reconstructed)
    x_seq_true, _, _ = model.slice_x(x_true)
    # TODO: think of a better way to do this...
    if all([hasattr(model, 'use_a'), hasattr(model, 'use_b'), hasattr(model, 'use_pep')]):
        x_b_recon, x_a_recon, x_pep_recon = x_seq_recon
        x_b_true, x_a_true, x_pep_true = x_seq_true
        if model.use_b:
            df['b_acc'] = metrics['b_accuracy']
            df['b_hat_reconstructed'] = model.recover_sequences_blosum(x_b_recon)
            df['b_true_reconstructed'] = model.recover_sequences_blosum(x_b_true)
            df['n_errors_b'] = df.apply(
                lambda x: sum([c1 != c2 for c1, c2 in zip(x['b_hat_reconstructed'], x['b_true_reconstructed'])]),
                axis=1)
        if model.use_a:
            df['a_acc'] = metrics['a_accuracy']
            df['a_hat_reconstructed'] = model.recover_sequences_blosum(x_a_recon)
            df['a_true_reconstructed'] = model.recover_sequences_blosum(x_a_true)
            df['n_errors_a'] = df.apply(
                lambda x: sum([c1 != c2 for c1, c2 in zip(x['a_hat_reconstructed'], x['a_true_reconstructed'])]),
                axis=1)
        if model.use_pep:
            df['pep_acc'] = metrics['pep_accuracy']
            df['pep_hat_reconstructed'] = model.recover_sequences_blosum(x_pep_recon)
            df['pep_true_reconstructed'] = model.recover_sequences_blosum(x_pep_true)
            df['n_errors_pep'] = df.apply(
                lambda x: sum([c1 != c2 for c1, c2 in zip(x['pep_hat_reconstructed'], x['pep_true_reconstructed'])]),
                axis=1)
    else:
        df['seq_acc'] = metrics['seq_accuracy']
        seq_hat_reconstructed = model.recover_sequences_blosum(x_seq_recon)
        seq_true_reconstructed = model.recover_sequences_blosum(x_seq_true)
        df['hat_reconstructed'] = seq_hat_reconstructed
        df['true_reconstructed'] = seq_true_reconstructed
        df['n_errors_seq'] = df.apply(
            lambda x: sum([c1 != c2 for c1, c2 in zip(x['hat_reconstructed'], x['true_reconstructed'])]), axis=1)

    if model.use_v:
        df['v_correct'] = metrics['v_accuracy']
    if model.use_j:
        df['j_correct'] = metrics['j_accuracy']

    z_latent = torch.cat(z_latent).detach()
    df[[f'z_{i}' for i in range(z_latent.shape[1])]] = z_latent
    return df


def train_eval_loops(n_epochs, tolerance, model, criterion, optimizer,
                     train_loader, valid_loader, checkpoint_filename, outdir):
    """ Trains and validates a model over n_epochs, then reloads the best checkpoint

    Args:
        n_epochs:
        tolerance:
        model:
        criterion:
        optimizer:
        train_loader:
        valid_loader:
        checkpoint_filename:
        outdir:

    Returns:
        model
        train_metrics
        valid_metrics
        train_losses
        valid_losses
        best_epoch
        best_val_loss
        best_val_auc
    """
    if not checkpoint_filename.endswith('.pt'):
        checkpoint_filename = checkpoint_filename + '.pt'
    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_reconstruction, best_epoch = 1000, 0., 1
    best_val_losses, best_val_metrics = {}, {}
    # To normalize the mean accuracy thing depending on the amount of different metrics we are using
    divider = int(model.use_v) + int(model.use_j)
    if all([hasattr(model, 'use_a'), hasattr(model, 'use_b'), hasattr(model, 'use_pep')]):
        divider += (int(model.use_b) + int(model.use_a) + int(model.use_pep))
    else:
        divider += 1
    best_dict = {}
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_model_step(model, criterion, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs + 1:
            train_loss_text = f'Train: Epoch {e}\nLoss:\tReconstruction: {train_loss["reconstruction"]:.4f}\tKLD: {train_loss["kld"]:.4f}'
            if 'triplet' in train_loss.keys():
                train_loss_text = train_loss_text + f'\tTriplet: {train_loss["triplet"]:.4f}'
            train_metrics_text = 'Accs:\t' + ',\t'.join(
                [f"{k.replace('accuracy', 'acc')}:{v:.2%}" for k, v in train_metric.items()])
            train_text = '\n'.join([train_loss_text, train_metrics_text])
            valid_loss_text = f'Valid: Epoch {e}\nLoss:\tReconstruction: {valid_loss["reconstruction"]:.4f}\tKLD: {valid_loss["kld"]:.4f}'
            if 'triplet' in valid_loss.keys():
                valid_loss_text = valid_loss_text + f'\tTriplet: {valid_loss["triplet"]:.4f}'
            valid_metrics_text = 'Accs:\t' + ',\t'.join(
                [f"{k.replace('accuracy', 'acc')}:{v:.2%}" for k, v in valid_metric.items()])
            valid_text = '\n'.join([valid_loss_text, valid_metrics_text])
            tqdm.write(train_text)
            tqdm.write(valid_text)
        # mean_accuracy = np.sum([valid_metric['seq_accuracy'], valid_metric['v_accuracy'], valid_metric['j_accuracy']]) / divider
        mean_accuracy = np.sum([x for x in valid_metric.values()]) / divider

        if e > 1 and ((valid_loss[
                           "total"] <= best_val_loss + tolerance and mean_accuracy > best_val_reconstruction) or mean_accuracy > best_val_reconstruction):
            # Getting the individual components for asserts
            best_epoch = e
            best_val_loss = valid_loss['total']
            best_val_reconstruction = mean_accuracy
            # Saving the actual dictionaries for logging purposes
            best_val_losses = valid_loss
            best_val_metrics = valid_metric

            best_dict = {'Best epoch': best_epoch}
            best_dict.update(valid_loss)
            best_dict.update(valid_metric)
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)

    print(f'End of training cycles')
    print(best_dict)
    best_train_reconstruction = max([np.sum([x for x in z.values()]) / divider for z in train_metrics])

    print(f'Best train loss:\t{min([x["total"] for x in train_losses]):.3e}'
          # f'Best train reconstruction Acc:\t{max([x["seq_accuracy"] for x in train_metrics])}')
          f'Best train reconstruction Acc:\t{best_train_reconstruction:.3%}')
    print(f'Best valid epoch: {best_epoch}')
    print(f'Best valid loss :\t{best_val_loss:.3e}, best valid mean reconstruction acc:\t{best_val_reconstruction:.3%}')
    # print(f'Reloaded best model at {os.path.abspath(os.path.join(outdir, checkpoint_filename))}')
    model = load_checkpoint(model, checkpoint_filename, outdir)
    # Here for now it's not the best implementation but save the full model with a undefined json filename
    model.eval()
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics


from src.datasets import TCRpMHCDataset
from src.metrics import get_metrics


def train_classifier_step(model, criterion, optimizer, train_loader):
    model.train()
    acum_loss = 0
    y_scores, y_true = [], []
    for x, y in train_loader:
        x, y = x.to(model.device), y.to(model.device)
        # output here should be logits to use BCEWithLogitLoss
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Saving scores and true for aucs etc
        y_scores.append(output.detach().cpu())
        y_true.append(y.detach().cpu())
        acum_loss += loss.item() * x.shape[0]

    acum_loss /= len(train_loader.dataset)
    # Saving the scores as logits but getting the metrics using sigmoid, shouldn't change much but makes the y_pred and thresholding easier
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    train_metrics = get_metrics(y_true, F.sigmoid(y_scores), threshold=0.5, reduced=True, round_digit=5)
    return acum_loss, train_metrics


def eval_classifier_step(model, criterion, valid_loader):
    model.train()
    acum_loss = 0
    y_scores, y_true = [], []
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(model.device), y.to(model.device)
            # output here should be logits to use BCEWithLogitLoss
            output = model(x)
            loss = criterion(output, y)
            # Saving scores and true for aucs etc
            y_scores.append(output.detach().cpu())
            y_true.append(y.detach().cpu())
            acum_loss += loss.item() * x.shape[0]

    acum_loss /= len(valid_loader.dataset)
    # Saving the scores as logits but getting the metrics using sigmoid, shouldn't change much but makes the y_pred and thresholding easier
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    valid_metrics = get_metrics(y_true, F.sigmoid(y_scores), threshold=0.5, reduced=True, round_digit=5)
    return acum_loss, valid_metrics


def predict_classifier(model, dataset, dataloader):
    assert type(
        dataloader.sampler) == torch.utils.data.SequentialSampler, 'Test/Valid loader MUST use SequentialSampler!'
    df = dataset.df.reset_index(drop=True).copy()
    y_true, y_logit = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)
            output = model(x)
            y_true.append(y.detach().cpu())
            y_logit.append(output.detach().cpu())

    y_logit, y_true = torch.cat(y_logit), torch.cat(y_true)
    y_probs = F.sigmoid(y_logit)

    df['pred_logit'] = y_logit
    df['pred_prob'] = y_probs

    return df


def classifier_train_eval_loops(n_epochs, tolerance, model, criterion, optimizer, train_loader, valid_loader,
                                checkpoint_filename, outdir):
    if not checkpoint_filename.endswith('.pt'):
        checkpoint_filename = checkpoint_filename + '.pt'
    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_auc, best_val_auc01, best_epoch = 1000, 0.5, 0.5, 1
    best_val_metrics = {}
    best_dict = {}
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_classifier_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_classifier_step(model, criterion, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % math.ceil(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs + 1:
            train_loss_text = f'Train: Epoch {e}\nLoss:: {train_loss:.4f}'
            train_metrics_text = ''.join([f'{k}:\t{v:.4f}' for k, v in train_metric.items() if
                                  k in ['auc', 'auc_01', 'accuracy', 'AP']])
            train_text = '\n'.join([train_loss_text, train_metrics_text])
            valid_loss_text = f'Valid: Epoch {e}\nLoss:: {valid_loss:.4f}'
            valid_metrics_text = ''.join([f'{k}:\t{v:.4f}' for k, v in valid_metric.items() if
                                  k in ['auc', 'auc_01', 'accuracy', 'AP']])
            valid_text = '\n'.join([valid_loss_text, valid_metrics_text])
            tqdm.write(train_text)
            tqdm.write(valid_text)

        if e > 1 and (valid_loss <= best_val_loss + tolerance and (
                valid_metric['auc'] >= (best_val_auc - tolerance) or valid_metric['auc_01'] >= (
                best_val_auc01 - tolerance))):
            best_epoch = e
            best_val_loss = valid_loss
            best_val_auc = valid_metric['auc']
            best_val_auc01 = valid_metric['auc_01']
            best_val_metrics = valid_metric
            # Saving best dict for logging purposes
            best_dict['epoch'] = best_epoch
            best_dict['loss'] = valid_loss
            best_dict.update(valid_metric)
            # Saving model
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir, best_dict=best_dict)

        print(f'End of training cycles')
        print(best_dict)

        print(f'Best train loss:\t{min(train_losses):.3e}, at epoch = {train_losses.index(min(train_losses))}'
              f'Train AUC, AUC01:\t{train_metrics[train_losses.index(min(train_losses))]["auc"]:.3%},\t{train_metrics[train_losses.index(min(train_losses))]["auc_01"]}')
        print(f'Best valid epoch: {best_epoch}')
        print(
            f'Best valid loss :\t{best_val_loss:.3e}, best valid AUC, AUC01:\t{best_val_auc:.3%},\t{best_val_auc01:.3%}')
        model = load_checkpoint(model, checkpoint_filename, outdir)
        model.eval()

    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_loss, best_val_metrics
