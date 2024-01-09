import argparse
import os
import pickle
import pandas as pd
from itertools import chain, cycle
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from sklearn.model_selection import KFold
import secrets
import string
from datetime import datetime as dt


def get_class_initcode_keys(class_, dict_kwargs):
    init_code = class_.__init__.__code__
    init_code = class_.__init__.__code__.co_varnames[1:init_code.co_argcount]
    return [x for x in dict_kwargs.keys() if x in init_code]


def get_dict_of_lists(list_of_dicts, name, filter=None):
    filter = list_of_dicts[0].keys() if filter is None else filter
    return {f'{name}_{key}': [d[key] for d in list_of_dicts] for key in list_of_dicts[0] if key in filter}


def get_motif(row, seq_col, window_size):
    return row[seq_col][int(row['core_start_index']):int(row['core_start_index']) + window_size]


def plot_loss_aucs(train_losses, valid_losses, train_aucs, valid_aucs,
                   filename, outdir, dpi=300, palette='gnuplot2'):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('loss')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('AUC')
    a[1].set_xlabel('epochs')
    f.savefig(f'{outdir}{filename}.png', dpi=dpi, bbox_inches='tight')


def plot_vae_loss_accs(losses_dict, accs_dict, filename, outdir, dpi=300,
                       palette='gnuplot2_r', warm_up=10,
                       figsize=(14, 10), ylim0=[0,1], ylim1=[0.5,1.1], title=None):
    """

    Args:
        train_losses: list of dictionaries for train_loss
        valid_losses: list of dictionaries for valid_loss
        train_accs: list of dictionaries for train_acc
        valid_accs: list of dictionaries for valid_acc
        filename:
        outdir:
        dpi:
        palette:

    Returns:

    """
    n = max(len(losses_dict.keys()), len(accs_dict.keys()))
    sns.set_palette(get_palette(palette, n_colors=n))
    f, a = plt.subplots(2, 1, figsize=figsize)
    a = a.ravel()
    # Corresponds to the warmup
    warm_up = 0 if warm_up is None else warm_up
    # plotting each component of the loss.
    # Should be 3 elements for each dict (total/recon/kld) and (seq/v/j)
    # Reformatting the list of dicts into dicts of lists:
    best_val_loss_epoch = -1
    best_val_accs_epoch = -1
    for k, v in losses_dict.items():
        if len(v) == 0 or all([val == 0 for val in v]): continue
        a[0].plot(v[warm_up:], label=k)
        if k == 'valid_total' or k=='valid_loss':
            best_val_loss_epoch = v.index(min(v))
    for k, v in accs_dict.items():
        if len(v) == 0 or all([val == 0 for val in v]): continue
        a[1].plot(v[warm_up:], label=k)
        if k == 'valid_seq_accuracy' or k == 'valid_b_accuracy' or k== 'valid_auc':
            best_val_accs_epoch = v.index(max(v))
    a[0].set_ylim(ylim0)
    a[0].axvline(x=best_val_loss_epoch, ymin=0, ymax=1, ls='--', lw=0.5,
                 c='k', label=f'Best loss epoch {best_val_loss_epoch}')
    a[1].axvline(x=best_val_accs_epoch, ymin=0, ymax=1, ls='--', lw=0.5,
                 c='k', label=f'Best accs epoch {best_val_accs_epoch}')
    a[1].set_ylim(ylim1)
    a[0].set_title('Losses')
    a[1].set_title('Accuracies')
    a[0].legend()
    a[1].legend()
    a[0].set_xlabel('epochs')
    a[1].set_xlabel('epochs')
    if title is not None:
        f.suptitle(title, fontweight='semibold', fontsize=14)
        f.tight_layout()
    f.savefig(f'{outdir}{filename}.png', dpi=dpi, bbox_inches='tight')
    return f,a


def get_datetime_string():
    return dt.now().strftime("%y%m%d_%H%M")


def get_random_id(length=6):
    first_character = ''.join(
        secrets.choice(string.digits) for _ in range(2))  # Generate a random digit for the first character
    remaining_characters = ''.join(
        secrets.choice(string.ascii_letters + string.digits) for _ in
        range(length - 2))  # Generate L-2 random characters
    random_string = first_character + remaining_characters
    return random_string


def make_chunks(iterable, chunk_size):
    k, m = divmod(len(iterable), chunk_size)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chunk_size))


def get_kfolds(df, k, xcol, ycol, shuffle=False, random_state=None):
    """ Splits & assigns the fold numbers
    Args:
        df:
        k:
        shuffle:
        random_state:

    Returns:
        df: df with column fold according to the Kfolds
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    df['fold'] = -1
    for i, (train_idx, test_idx) in enumerate(kf.split(df[xcol].values, df[ycol])):
        df.iloc[test_idx, df.columns.get_loc('fold')] = i
    df.fold = df.fold.astype(int)
    return df


def get_palette(palette, n_colors):
    """ 'stretches' stupid fucking palette to have more contrast"""
    if n_colors == 2:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[-1]]
    elif n_colors == 3:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[2], pal[-1]]
    else:
        nc = int(n_colors * 2)
        pal = sns.color_palette(palette, n_colors=nc)
        palette = [pal[i] for i in range(1, 1 + int(n_colors * 2), 2)]
    return palette


def convert_hla(hla):
    if not hla.startswith('HLA-'):
        hla = 'HLA-' + hla
    return hla.replace('*', '').replace(':', '')


def add_median_labels(ax, fmt='.1%'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def flatten_product(container):
    """
    Flattens a product or container into a flat list, useful when product/chaining many conditions
    Looks into each sub-element & recursively calls itself
    Args:
        container:
    Returns:

    """
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten_product(i):
                yield j
        else:
            yield i


def str2bool(v):
    """converts str to bool from argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pkl_dump(obj, filename, dirname=None):
    if dirname is not None:
        mkdirs(dirname)
        filename = os.path.join(dirname, filename)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print(f'{os.path.abspath(filename)} saved.')


def pkl_load(filename, dirname=None):
    if dirname is not None:
        filename = os.path.join(dirname, filename)
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        raise ValueError(f'Unable to load or find {os.path.abspath(os.path.join(dirname, filename))}!')


def flatten_level_columns(df: pd.DataFrame, levels=[0, 1]):
    df.columns = [f'{x.lower()}_{y.lower()}'
                  for x, y in zip(df.columns.get_level_values(levels[0]),
                                  df.columns.get_level_values(levels[1]))]
    return df


def convert_path(path):
    return path.replace('\\', '/')


def get_plot_corr(df, cols, which='spearman', title='', figsize=(13, 12.5), palette='viridis'):
    corr = df[cols].corr(which)
    f, a = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(corr.round(2), center=0, xticklabels=corr.columns, yticklabels=corr.columns,
                cmap=palette, vmax=1, vmin=-1, annot=True, square=True, annot_kws={'weight': 'semibold'})
    a.set_xticklabels(a.get_xticklabels(), rotation=30, fontweight='semibold')
    a.set_yticklabels(a.get_yticklabels(), fontweight='semibold')
