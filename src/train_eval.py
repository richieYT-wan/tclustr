import copy
import multiprocessing
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import wandb
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.data_processing import encoding_matrix_dict
from src.torch_utils import save_checkpoint, load_checkpoint
from src.metrics import get_metrics, reconstruction_accuracy


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


# TODO : HERE CHANGE ALL THE CODE TO ACCOMODATE FOR VAE AND THE SPLIT OPTIMIZER AS WELL AS SPLIT LOSSES WITH THE CUSTOM LOSS!!1
# TODO: STOP BEING A LAZY FUCK AND JUST FIX THIS JFC
# Maybe no need to do the split optimizer because the loss behaviour seemed to come from the weight decay and not the split LR

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
    acum_total_loss, acum_recon_loss, acum_kld_loss = 0, 0, 0
    x_reconstructed, x_true = [], []

    for x in train_loader:
        x_hat, mu, logvar = model(x)
        recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
        loss = recon_loss + kld_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x_reconstructed.append(x_hat)
        x_true.append(x)
        acum_total_loss += loss.item() * x.shape[0]
        acum_recon_loss += recon_loss.item() * x.shape[0]
        acum_kld_loss += kld_loss.item() * x.shape[0]

    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)
    seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, v_true, j_true = model.reconstruct_hat(x_true)

    seq_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat)
    # Normalizes to loss per batch
    acum_total_loss /= len(train_loader.dataset)
    acum_recon_loss /= len(train_loader.dataset)
    acum_kld_loss /= len(train_loader.dataset)
    train_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}
    train_metrics = {'seq_accuracy': seq_accuracy, 'v_accuracy': v_accuracy, 'j_accuracy': j_accuracy}
    return train_loss, train_metrics


def eval_model_step(model, criterion, valid_loader):
    model.eval()
    # disables gradient logging
    acum_total_loss, acum_recon_loss, acum_kld_loss = 0, 0, 0
    x_reconstructed, x_true = [], []
    with torch.no_grad():
        # Same workaround as above
        for x in valid_loader:
            x_hat, mu, logvar = model(x)
            recon_loss, kld_loss = criterion(x_hat, x, mu, logvar)
            loss = recon_loss + kld_loss
            x_reconstructed.append(x_hat)
            x_true.append(x)
            acum_total_loss += loss.item() * x.shape[0]
            acum_recon_loss += recon_loss.item() * x.shape[0]
            acum_kld_loss += kld_loss.item() * x.shape[0]
    # Concatenate the y_pred & y_true tensors and compute metrics
    x_reconstructed, x_true = torch.cat(x_reconstructed), torch.cat(x_true)

    seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, v_true, j_true = model.reconstruct_hat(x_true)

    seq_accuracy, v_accuracy, j_accuracy = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat)
    # Normalizes to loss per batch
    acum_total_loss /= len(valid_loader.dataset)
    acum_recon_loss /= len(valid_loader.dataset)
    acum_kld_loss /= len(valid_loader.dataset)
    valid_loss = {'total': acum_total_loss, 'reconstruction': acum_recon_loss, 'kld': acum_kld_loss}
    valid_metrics = {'seq_accuracy': seq_accuracy, 'v_accuracy': v_accuracy, 'j_accuracy': j_accuracy}
    return valid_loss, valid_metrics


def predict_model(model, dataset: torch.utils.data.Dataset, dataloader: torch.utils.data.DataLoader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'
    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'
    model.eval()

    cols_to_keep = ['B3', 'TRBV_gene', 'TRBJ_gene', 'peptide']
    df = dataset.df.reset_index(drop=True).copy()
    df = df[[x for x in cols_to_keep if x in df.columns]]
    x_reconstructed, x_true, z_latent = [], [], []
    with torch.no_grad():
        # Same workaround as above
        for x in dataloader:
            x_hat, _, _ = model(x)
            z = model.embed(x)
            x_reconstructed.append(x_hat)
            x_true.append(x)
            z_latent.append(z)

    x_reconstructed = torch.cat(x_reconstructed)
    x_true = torch.cat(x_true)
    seq_hat, v_hat, j_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, v_true, j_true = model.reconstruct_hat(x_true)
    seq_acc, v_acc, j_acc = reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat, return_per_element=True)

    x_seq_recon, _, _ = model.slice_x(x_reconstructed)
    x_seq_true, _, _ = model.slice_x(x_true)

    seq_hat_reconstructed = model.recover_sequences_blosum(x_seq_recon)
    seq_true_reconstructed = model.recover_sequences_blosum(x_seq_true)
    df['hat_reconstructed'] = seq_hat_reconstructed
    df['true_reconstructed'] = seq_true_reconstructed
    df['seq_acc'] = seq_acc
    df['v_correct'] = v_acc
    df['j_correct'] = j_acc
    z_latent = torch.cat(z_latent).detach()
    df[[f'z_{i}' for i in range(z_latent.shape[1])]] = z_latent
    return df


def train_eval_loops(n_epochs, tolerance, model, criterion, optimizer,
                     train_dataset, train_loader, valid_loader,
                     checkpoint_filename, outdir):
    """ Trains and validates a model over n_epochs, then reloads the best checkpoint


    Args:
        n_epochs:
        tolerance:
        model:
        criterion:
        optimizer:
        train_dataset: Torch Dataset, is here so we can do standardize & burn-in periods
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

    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_reconstruction, best_epoch = 1000, 0., 1
    best_val_losses, best_val_metrics = {}, {}
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_model_step(model, criterion, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (n_epochs >= 10 and e % int(0.05 * n_epochs) == 0) or e == 1 or e == n_epochs+1:
            tqdm.write(
                f'\nEpoch {e}: train recon loss, train kld loss, seq_acc, v_acc, j_acc:\t{train_loss["reconstruction"]:.4f},\t{train_loss["kld"]:.4f}\t{train_metric["seq_accuracy"]:.3f}, \t{train_metric["v_accuracy"]:.3f}, \t{train_metric["j_accuracy"]:.3f}')
            tqdm.write(
                f'Epoch {e}: valid recon loss, train kld loss, seq_acc, v_acc, j_acc:\t{valid_loss["reconstruction"]:.4f},\t{valid_loss["kld"]:.4f}\t{valid_metric["seq_accuracy"]:.3f}, \t{valid_metric["v_accuracy"]:.3f}, \t{valid_metric["j_accuracy"]:.3f}')

        # Doesn't allow saving the very first model as sometimes it gets stuck in a random state that has good.
        # Here, will take the best overall reconstruction (mean for seq, V, and J because V reconstruction somehow gets stuck at some low values
        mean_accuracy = np.mean([valid_metric['seq_accuracy'], valid_metric['v_accuracy'], valid_metric['j_accuracy']])
        if e > 1 and ((valid_loss["total"] <= best_val_loss + tolerance and mean_accuracy > best_val_reconstruction) \
                      or mean_accuracy > best_val_reconstruction):
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
    print(f'Best train loss:\t{min([x["total"] for x in train_losses]):.3e}'
          f'Best train AUC:\t{max([x["seq_accuracy"] for x in train_metrics])}')
    print(f'Best valid epoch: {best_epoch}')
    print(f'Best valid loss :\t{best_val_loss:.3e}, best valid mean reconstruction acc:\t{best_val_reconstruction}')
    # print(f'Reloaded best model at {os.path.abspath(os.path.join(outdir, checkpoint_filename))}')
    model = load_checkpoint(model, checkpoint_filename, outdir)
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_losses, best_val_metrics
