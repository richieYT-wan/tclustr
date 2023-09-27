import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn as nn

from src.data_processing import verify_df, get_dataset, encoding_matrix_dict
from src.utils import get_palette
from torch import nn
from torch.nn import functional as F

mpl.rcParams['figure.dpi'] = 180
sns.set_style('darkgrid')
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, \
    recall_score, precision_score, precision_recall_curve, auc, average_precision_score


class VAELoss(nn.Module):
    """
    No fucking annealing, just some basic stuff for now
    TODO: re-do the tanh behaviour for the KLD loss
    """

    def __init__(self, sequence_criterion=nn.MSELoss(reduction='mean'),
                 use_v=True, use_j=True, max_len=21, aa_dim=20, v_dim=51, j_dim=13,
                 weight_seq=1, weight_v=.3, weight_j=.15, weight_kld=.5, debug=False, warm_up=True, n_batches=67):
        super(VAELoss, self).__init__()
        weight_v = weight_v if use_v else 0
        weight_j = weight_j if use_j else 0
        self.norm_factor = weight_seq + weight_v + weight_j + weight_kld
        self.sequence_criterion = sequence_criterion
        self.max_len = max_len
        self.aa_dim = aa_dim
        self.use_v = use_v
        self.use_j = use_j
        self.v_dim = v_dim if use_v else 0
        self.j_dim = j_dim if use_j else 0
        self.weight_seq = weight_seq / self.norm_factor
        self.weight_v = weight_v / self.norm_factor
        self.weight_j = weight_j / self.norm_factor
        self.base_weight_kld = weight_kld / self.norm_factor
        self.weight_kld = self.base_weight_kld
        self.step = 0
        self.debug = debug
        self.warm_up = warm_up
        self.n_batches = n_batches
        print(self.weight_seq, self.weight_v, self.weight_j, self.base_weight_kld)

    def slice_x(self, x):
        sequence = x[:, 0:(self.max_len * self.aa_dim)].view(-1, self.max_len, self.aa_dim)
        # Reconstructs the v/j gene as one hot vectors
        v_gene = x[:, (self.max_len * self.aa_dim):(self.max_len * self.aa_dim + self.v_dim)] if self.use_v else None
        j_gene = x[:, ((self.max_len * self.aa_dim) + self.v_dim):] if self.use_j else None
        return sequence, v_gene, j_gene

    def forward(self, x_hat, x, mu, logvar):

        x_hat_seq, x_hat_v, x_hat_j = self.slice_x(x_hat)
        x_true_seq, x_true_v, x_true_j = self.slice_x(x)
        reconstruction_loss = self.weight_seq * self.sequence_criterion(x_hat_seq, x_true_seq)
        if self.debug:
            print('seq_loss', reconstruction_loss)
        if self.use_v:
            # Something wrong here with_loss, it explodes to minus infinity for some reasons
            v_loss = self.weight_v * F.cross_entropy(x_hat_v, x_true_v.argmax(dim=1), reduction='mean')
            if self.debug:
                print('v_loss', v_loss)
            reconstruction_loss += v_loss
        if self.use_j:
            j_loss = self.weight_j * F.cross_entropy(x_hat_j, x_true_j.argmax(dim=1), reduction='mean')
            if self.debug:
                print('j_loss', j_loss)
            reconstruction_loss += j_loss

        if self.step <= 500 and not self.warm_up:
            self.weight_kld = self.step / 1000 * self.base_weight_kld
        else:
            if self.warm_up and self.step <= self.n_batches * 15:
                self.weight_kld = 0

        kld = self.weight_kld * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        self.step += 1
        if self.step >= self.n_batches*15:
            self.warm_up=False
        if not self.warm_up:
            self.weight_kld = max(self.base_weight_kld - (self.base_weight_kld * (self.step / 1000)),
                              self.base_weight_kld / 5)
        if self.debug:
            print('kld_weight', self.weight_kld)
            print('kld_loss', kld, '\n')

        # Return them separately and sum later so that I can debug each component
        return reconstruction_loss, kld

    def set_debug(self, debug):
        self.debug = debug

    def reset_parameters(self):
        self.step = 0


# NOTE : Fixed version with the true acc
def reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat, pad_index=20, return_per_element=False):
    """

    Args:
        seq_true: ordinal vector of true sequence (Here, the number is to map back to an amino acid with AA_KEYS, with 20 = X)
        seq_hat: Same but reconstructed
        v_true: same but for true V
        v_hat: same for recons V
        j_true: ...
        j_hat: ...
        pad_index: This is the INDEX for the AA used for padding. 20 for default (which represents X)
        return_per_element: Return a per-element acc instead of mean

    Returns:

    """
    # Compute the mask and true lengths
    mask = (seq_true != pad_index).float()
    true_lens = mask.sum(dim=1)
    # difference here for per element is that we don't take the mean(dim=0) and have to detach() from graph to do tolist()
    seq_accuracy = ((seq_true == seq_hat).float() * mask).sum(1) / true_lens
    if return_per_element:
        seq_accuracy = seq_accuracy.detach().cpu().tolist()
        v_accuracy = ((v_true.argmax(dim=1) == v_hat.argmax(dim=1)).float()).detach().cpu().int().tolist() if v_hat is not None else 0
        j_accuracy = ((j_true.argmax(dim=1) == j_hat.argmax(dim=1)).float()).detach().cpu().int().tolist() if j_hat is not None else 0
    else:
        seq_accuracy = seq_accuracy.mean(dim=0).item()

        v_accuracy = ((v_true.argmax(dim=1) == v_hat.argmax(dim=1)).float().mean(dim=0)).item() if v_hat is not None else 0
        j_accuracy = ((j_true.argmax(dim=1) == j_hat.argmax(dim=1)).float().mean(dim=0)).item() if j_hat is not None else 0
    return seq_accuracy, v_accuracy, j_accuracy

# # NOTE: OLD VERSION WITH THE PADDING IN SEQ_ACC
# def reconstruction_accuracy(seq_true, seq_hat, v_true, v_hat, j_true, j_hat,
#                             return_per_element=False):
#     # todo: Do accuracy with and without padding
#     if return_per_element:
#         seq_accuracy = ((seq_true == seq_hat).float().mean(dim=1)).detach().cpu().tolist()
#         v_accuracy = ((v_true.argmax(dim=1) == v_hat.argmax(dim=1)).float()).detach().cpu().int().tolist()
#         j_accuracy = ((j_true.argmax(dim=1) == j_hat.argmax(dim=1)).float()).detach().cpu().int().tolist()
#     else:
#         seq_accuracy = ((seq_true == seq_hat).float().mean(dim=1).mean(dim=0)).item()
#         v_accuracy = ((v_true.argmax(dim=1) == v_hat.argmax(dim=1)).float().mean(dim=0)).item()
#         j_accuracy = ((j_true.argmax(dim=1) == j_hat.argmax(dim=1)).float().mean(dim=0)).item()
#     return seq_accuracy, v_accuracy, j_accuracy





def auc01_score(y_true: np.ndarray, y_pred: np.ndarray, max_fpr=0.1) -> float:
    """Compute the partial AUC of the ROC curve for FPR up to max_fpr.
    Args:
        y_true (array-like): The true labels of the data (0 or 1).
        y_pred (array-like): The predicted probabilities or scores.
        max_fpr (float): Maximum false positive rate.
    Returns:
        float: Partial AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return auc(fpr, tpr) * 10


def get_metrics(y_true, y_score, y_pred=None, threshold=0.50, keep=False, reduced=True, round_digit=4):
    """
    Computes all classification metrics & returns a dictionary containing the various key/metrics
    incl. ROC curve, AUC, AUC_01, F1 score, Accuracy, Recall
    Args:
        y_true:
        y_pred:
        y_score:

    Returns:
        metrics (dict): Dictionary containing all results
    """
    metrics = {}
    # DETACH & PASS EVERYTHING TO CPU
    if threshold is not None and y_pred is None:
        # If no y_pred is provided, will threshold score (y in [0, 1])
        y_pred = (y_score > threshold)
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy()
        elif type(y_pred) == np.ndarray:
            y_pred = y_pred.astype(int)
    elif y_pred is not None and type(y_pred) == torch.Tensor:
        y_pred = y_pred.int().cpu().detach().numpy()

    if type(y_true) == torch.Tensor and type(y_score) == torch.Tensor:
        y_true, y_score = y_true.int().cpu().detach().numpy(), y_score.cpu().detach().numpy()

    metrics['auc'] = roc_auc_score(y_true, y_score)
    metrics['auc_01_std'] = roc_auc_score(y_true, y_score, max_fpr=0.1)
    metrics['auc_01'] = auc01_score(y_true, y_score, max_fpr=0.1)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AP'] = average_precision_score(y_true, y_score)
    if not reduced:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics['roc_curve'] = fpr, tpr
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['pr_curve'] = recall, precision  # So it follows the same x,y format as roc_curve
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
            metrics['prauc'] = auc(recall, precision)
            metrics['AP'] = average_precision_score(y_true, y_score)
        except:
            print('Couldn\'t get AUCs/etc because there\'s only one class in the dataset')
            print(f'Only negatives: {all(y_true == 0)}, Only positives: {all(y_true == 1)}')
            raise ValueError
        if keep:
            metrics['y_true'] = y_true
            metrics['y_score'] = y_score
    if round_digit is not None:
        for k, v in metrics.items():
            metrics[k] = round(v, round_digit)
    return metrics


def plot_roc_auc_fold(results_dict, palette='hsv', n_colors=None, fig=None, ax=None,
                      title='ROC AUC plot\nPerformance for average prediction from models of each fold',
                      bbox_to_anchor=(0.9, -0.1)):
    n_colors = len(results_dict.keys()) if n_colors is None else n_colors
    sns.set_palette(palette, n_colors=n_colors)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    print(results_dict.keys())
    for k in results_dict:
        if k == 'kwargs': continue
        fpr = results_dict[k]['roc_curve'][0]
        tpr = results_dict[k]['roc_curve'][1]
        auc = results_dict[k]['auc']
        auc_01 = results_dict[k]['auc_01']
        # print(k, auc, auc_01)
        style = '--' if type(k) == np.int32 else '-'
        alpha = 0.75 if type(k) == np.int32 else .9
        lw = .8 if type(k) == np.int32 else 1.5
        sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'{k}, AUC={auc.round(4)}, AUC_01={auc_01.round(4)}',
                     n_boot=50, ls=style, lw=lw, alpha=alpha)

    sns.lineplot([0, 1], [0, 1], ax=ax, ls='--', color='k', label='random', lw=0.5)
    if bbox_to_anchor is not None:
        ax.legend(bbox_to_anchor=bbox_to_anchor)

    ax.set_title(f'{title}')
    return fig, ax


def get_mean_roc_curve(roc_curves, extra_key=None, auc01=False):
    """
    Assumes a single-level dict, i.e. roc_curves_dict has all the outer folds, and no inner folds
    Or it is the sub-dict that contains all the inner folds for a given outer fold.
    i.e. to access a given fold's curve, should use `roc_curves_dict[number]['roc_curve']`
    Args:
        roc_curves_dict:
        extra_key (str) : Extra_key in case it's nested, like train_metrics[fold]['valid']['roc_curve']
    Returns:
        base_fpr
        mean_curve
        low_std_curve
        high_std_curve
        auc
    """

    # Base fpr to interpolate
    tprs = []
    aucs = []
    aucs_01 = []
    if type(roc_curves) == dict:
        if extra_key is not None:
            max_n = max([len(v[extra_key]['roc_curve'][0]) for k, v in roc_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_fpr = np.linspace(0, 1, max_n)
            for k, v in roc_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                fpr = v[extra_key]['roc_curve'][0]
                tpr = v[extra_key]['roc_curve'][1]
                # Interp TPR so it fits the right shape for base_fpr
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                tprs.append(tpr)
                aucs.append(v[extra_key]['auc'])
        else:
            max_n = max([len(v['roc_curve'][0]) for k, v in roc_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_fpr = np.linspace(0, 1, max_n)

            for k, v in roc_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                fpr = v['roc_curve'][0]
                tpr = v['roc_curve'][1]
                # Interp TPR so it fits the right shape for base_fpr
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                tprs.append(tpr)
                aucs.append(v['auc'])

    elif type(roc_curves) == list:
        # THIS HERE ASSUMES THE RESULTS ARE IN FORMAT [((fpr, tpr), auc) ...]
        max_n = max([len(x[0][0]) for x in roc_curves])
        base_fpr = np.linspace(0, 1, max_n)

        for curves in roc_curves:
            fpr = curves[0][0]
            tpr = curves[0][1]
            # Interp TPR so it fits the right shape for base_fpr
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            tprs.append(tpr)
            aucs.append(curves[1])
            if auc01:
                aucs_01.append(curves[-1])

    mean_auc = np.mean(aucs)
    if auc01:
        mean_auc01 = np.mean(aucs_01)

    tprs = np.stack(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    upper = np.minimum(mean_tprs + std_tprs, 1)
    lower = mean_tprs - std_tprs

    if auc01:
        return base_fpr, mean_tprs, lower, upper, mean_auc, mean_auc01
    else:
        return base_fpr, mean_tprs, lower, upper, mean_auc


def get_mean_pr_curve(pr_curves, extra_key=None):
    """
    Assumes a single-level dict, i.e. roc_curves_dict has all the outer folds, and no inner folds
    Or it is the sub-dict that contains all the inner folds for a given outer fold.
    i.e. to access a given fold's curve, should use `roc_curves_dict[number]['roc_curve']`
    Args:
        roc_curves_dict:
        extra_key (str) : Extra_key in case it's nested, like train_metrics[fold]['valid']['roc_curve']
    Returns:
        base_recall
        mean_curve
        std_curve
    """

    # Base recall to interpolate
    precisions = []
    aucs = []
    if type(pr_curves) == dict:
        if extra_key is not None:
            max_n = max([len(v[extra_key]['pr_curve'][0]) for k, v in pr_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_recall = np.linspace(0, 1, max_n)
            for k, v in pr_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                recall = v[extra_key]['pr_curve'][0]
                precision = v[extra_key]['pr_curve'][1]
                # Interp precision so it fits the right shape for base_recall
                precision = np.interp(base_recall, recall, precision)
                precision[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                precisions.append(precision)
                aucs.append(v[extra_key]['auc'])
        else:
            max_n = max([len(v['pr_curve'][0]) for k, v in pr_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_recall = np.linspace(0, 1, max_n)

            for k, v in pr_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                recall = v['pr_curve'][0]
                precision = v['pr_curve'][1]
                # Interp precision so it fits the right shape for base_recall
                precision = np.interp(base_recall, recall, precision)
                precision[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                precisions.append(precision)
                aucs.append(v['auc'])

    elif type(pr_curves) == list:
        # TODO FIX
        # THIS HERE ASSUMES THE RESULTS ARE IN FORMAT [((recall, precision), auc) ...]
        max_n = max([len(x[0][0]) for x in pr_curves])
        base_recall = np.linspace(0, 1, max_n)

        for curves in pr_curves:
            recall = curves[0][0]
            precision = curves[0][1]
            # Interp precision so it fits the right shape for base_recall
            precision = np.interp(base_recall, recall, precision)
            precision[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            precisions.append(precision)
            aucs.append(curves[1])

    mean_auc = np.mean(aucs)
    precisions = np.stack(precisions)
    mean_precisions = precisions.mean(axis=0)
    std_precisions = precisions.std(axis=0)
    upper = np.minimum(mean_precisions + std_precisions, 1)
    lower = mean_precisions - std_precisions
    return base_recall, mean_precisions, lower, upper, mean_auc


def get_roc(df, score='pred', target='agg_label', binder=None, anchor_mutation=None):
    """
    Args:
        df: DF containing the prediction or scores
        score: Name of the score columns, 'pred' by default
        target: Name of the target column, 'pred' by default
        binder: None, "Improved" or "Conserved" ; None by default
        anchor_mutation: None, True, False ; None by default

    Returns:

    """
    if binder is not None and anchor_mutation is not None:
        df = df.query('binder==@binder and anchor_mutation==@anchor_mutation').copy()
    try:
        fpr, tpr, _ = roc_curve(df[target].values, df[score].values)
        auc = roc_auc_score(df[target].values, df[score].values)
        auc01 = roc_auc_score(df[target].values, df[score].values, max_fpr=0.1)
    except KeyError:
        print('here')
        try:
            fpr, tpr, _ = roc_curve(df[target].values, df['mean_pred'].values)
            auc = roc_auc_score(df[target].values, df['mean_pred'].values)
            auc01 = roc_auc_score(df[target].values, df['mean_pred'].values, max_fpr=0.1)
        except:
            raise KeyError(f'{target} or "mean_pred" not in df\'s columns!')
    output = {"roc": (fpr, tpr),
              "auc": auc,
              "auc01": auc01,
              "npep": len(df)}
    return output
