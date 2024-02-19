import glob
import os
import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import math
import numpy as np
import torch
from torch import optim
from torch import cuda
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists, get_class_initcode_keys
from src.torch_utils import load_checkpoint, save_model_full, load_model_full
from src.models import FullTCRVAE
from src.train_eval import predict_model
from src.datasets import FullTCRDataset
from src.metrics import TwoStageVAELoss, compute_cosine_distance
from sklearn.metrics import roc_auc_score, precision_score
import argparse
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import hmean
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from functools import partial

def read_args(fn):
    with open(fn, 'r') as file:
        content = file.read()

    # Split the content into lines
    lines = content.strip().split('\n')

    # Initialize an empty dictionary to store the key-value pairs
    args = {}

    # Iterate through lines and extract key-value pairs
    for line in lines:
        if ':\t' in line:
            key, value = line.split(':\t')
        elif ': ' in line and line.count(': ') == 1:
            key, value = line.split(': ')
        else:
            continue
        # Convert numeric values to int or float
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                # Leave it as a string if conversion fails
                pass
        args[key.strip()] = value
    res = {}
    res['n_latent'] = args['latent_dim']
    res['w_kld'] = args['weight_kld']
    res['w_trp'] = args['weight_triplet'] if 'weight_triplet' in args else 'N/A'
    res['trp_type'] = args['dist_type'] if 'dist_type' in args else 'N/A'
    if res['w_trp'] == 0:
        res['trp_type'] = 'NoTriplet'
    res['margin'] = args['margin'] if 'margin' in args else 'N/A'
    return res


def get_cluster_summary(input_df, cluster='', label='label', feature='TSNE_1', kf=True):
    if kf:
        summaries = []
        dfs = []
        for fold in input_df.partition.unique():
            df = input_df.query('partition==@fold')
            df = df.groupby([cluster, label]).agg(count=(feature, 'count')).reset_index()
            summary = df.loc[df.groupby([cluster])['count'].idxmax(), [cluster, label]]
            cluster_sizes = df.groupby([cluster]).agg(cluster_size=('count', 'sum'))
            df = df.set_index([cluster, label])
            df['intra_cluster_percent'] = df['count'] / cluster_sizes['cluster_size'] * 100
            summary['purity_percent'] = summary.apply(lambda x: df.loc[x[cluster], x[label]]['intra_cluster_percent'],
                                                      axis=1)
            summary = pd.merge(summary, cluster_sizes.reset_index(), left_on=[cluster], right_on=[cluster])
            df.reset_index([label], inplace=True)
            df['cluster_size'] = summary['cluster_size']
            df['cluster_label'] = summary[label]
            df.reset_index(inplace=True)
            df.set_index([cluster, label], inplace=True)
            summary['partition'] = fold
            df['partition'] = fold
            summaries.append(summary)
            dfs.append(df)
        return pd.concat(summaries), pd.concat(dfs)
    else:
        df = input_df
        df = df.groupby([cluster, label]).agg(count=(feature, 'count')).reset_index()
        summary = df.loc[df.groupby([cluster])['count'].idxmax(), [cluster, label]]
        cluster_sizes = df.groupby([cluster]).agg(cluster_size=('count', 'sum'))
        df = df.set_index([cluster, label])
        df['intra_cluster_percent'] = df['count'] / cluster_sizes['cluster_size'] * 100
        summary['purity_percent'] = summary.apply(lambda x: df.loc[x[cluster], x[label]]['intra_cluster_percent'],
                                                  axis=1)
        summary = pd.merge(summary, cluster_sizes.reset_index(), left_on=[cluster], right_on=[cluster])
        df.reset_index([label], inplace=True)
        df['cluster_size'] = summary['cluster_size']
        df['cluster_label'] = summary[label]
        df.reset_index(inplace=True)
        df.set_index([cluster, label], inplace=True)
        summary['partition'] = 'ALL'
        df['partition'] = 'ALL'
        return summary, df


def get_clustering_stats(summary_df, percent=75, size=3, dict=False, return_df=True):
    tmp = summary_df.query('purity_percent>@percent and cluster_size>=@size')
    results = {'n_clusters': len(tmp),
               'n_total_tcrs': tmp.cluster_size.sum(),
               'mean_cluster_size': tmp.cluster_size.mean(),
               'n_singletons': len(summary_df.query('cluster_size==1')),
               'mean_purity': tmp.purity_percent.mean(),
               'max_purity': tmp.purity_percent.max(),
               'med_purity': tmp.purity_percent.median(),
               'std_purity': tmp.purity_percent.std(),
               'retention': tmp.cluster_size.sum() / summary_df.cluster_size.sum()}
    if dict:
        return results
    elif return_df:
        return pd.DataFrame(results, index=[0])
    else:
        return list(results.values())


def get_model(folder):
    pt = glob.glob(folder + '/*.pt')[0]
    js = glob.glob(folder + '/*.json')[0]
    model = load_model_full(pt, js, return_json=False, verbose=False)
    return model


def compute_distance(z, type='cos'):
    assert type in ['cos', 'l2', 'l1'], f'wrong dist type {type} ; must be in cos, l2, l1'
    distfct = compute_cosine_distance if 'cos' in type else torch.cdist
    p = 2 if type == 'l2' else 1
    return distfct(z, z, p=p)


def get_dist_matrix(preds, fold, distances=None, dist_type='cos'):
    """
        if distances is not None then it's precomputed and don't have to recompute each time
    """
    preds = preds.copy(deep=True)
    if distances is None:
        z_cols = [x for x in preds.columns if x.startswith('z_')]
        z = torch.from_numpy(preds[z_cols].values)
        distances = compute_distance(z, type=dist_type)

    preds['model_fold'] = fold
    preds['set'] = preds.apply(lambda x: 'train' if x['model_fold'] != x['partition'] else 'valid', axis=1)
    dist_matrix = pd.DataFrame(distances,
                               columns=[x['A3'] + '-' + x['B3'] for _, x in preds.iterrows()],
                               index=[x['A3'] + '-' + x['B3'] for _, x in preds.iterrows()])
    dist_matrix['set'] = preds['set'].values
    dist_matrix['label'] = preds['peptide'].values
    dist_matrix['binder'] = preds['binder'].values
    dist_matrix['origin'] = preds['origin'].values
    return dist_matrix


def get_dist_preds_summary(name, args, preds, distances, threshold, k, purity_percent, cluster_size, dist_type='cos'):
    preds = preds.query('model_fold==@k').copy(deep=True)
    dist = get_dist_matrix(preds, k, distances, dist_type)
    dist['partition'] = preds.partition.values
    dist['feature'] = preds.z_0.values
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete',
                                         distance_threshold=threshold)

    labcols = ['set', 'label', 'binder', 'origin']
    dropcols = labcols + [x for x in dist.columns if 'pred_' in x] + ['partition', 'feature']
    pred_clusters = clustering.fit_predict(dist.drop(columns=dropcols).values)
    dist[f'pred_cl_{threshold:.4f}'] = pred_clusters
    preds[f'pred_cl_{threshold:.4f}'] = pred_clusters
    preds['model_fold'] = k
    summary, _ = get_cluster_summary(dist, f'pred_cl_{threshold:.4f}', 'label', 'feature', True)
    summary['model_type'] = name
    summary['dist_type'] = dist_type
    summary['threshold'] = threshold
    summary['model_fold'] = k

    preds['set'] = preds.apply(lambda x: 'train' if x['model_fold'] != x['partition'] else 'valid', axis=1)
    summary['set'] = summary.apply(lambda x: 'train' if x['model_fold'] != x['partition'] else 'valid', axis=1)

    stats = pd.concat([get_clustering_stats(summary.query('set=="valid"'), purity_percent, cluster_size,
                                            return_df=True).assign(fold=k, set='valid', **args),
                       get_clustering_stats(summary.query('set=="train"'), purity_percent, cluster_size,
                                            return_df=True).assign(fold=k, set='train', **args)])
    # Here disable returning the preds for now
    # return preds, summary, stats
    return preds, summary, stats


def wrapper(fdir, fold, name, args, dist_type, dataset, loader):
    model = get_model(fdir)
    predictions = predict_model(model, dataset, loader).query('binder==1')
    predictions['model_fold'] = fold
    predictions['set'] = predictions.apply(lambda x: 'train' if x['model_fold'] != x['partition'] else 'valid', axis=1)
    z = torch.from_numpy(predictions[[x for x in predictions.columns if x.startswith('z_')]].values)
    distances = compute_distance(z, type=dist_type)
    dist_matrix = pd.DataFrame(distances,
                               columns=[x['A3'] + '-' + x['B3'] for _, x in predictions.iterrows()],
                               index=[x['A3'] + '-' + x['B3'] for _, x in predictions.iterrows()])
    dist_matrix['model_fold'] = predictions['model_fold'].values
    dist_matrix['partition'] = predictions['partition'].values
    dist_matrix['set'] = predictions['set'].values
    dist_matrix['label'] = predictions['peptide'].values
    dist_matrix['binder'] = predictions['binder'].values
    dist_matrix['origin'] = predictions['origin'].values
    thresholds = np.linspace(0.15, 0.9, 50).round(4) if dist_type == 'cos' else np.linspace(5, 50, 50)
    summaries, stats = [], []

    for t in thresholds:
        _, summary, stat = get_dist_preds_summary(name, args, predictions, distances, threshold=t, k=fold,
                                                  purity_percent=70, cluster_size=3, dist_type=dist_type)
        stats.append(stat.assign(name=name, threshold=t))
        summaries.append(summary.assign(name=name, threshold=t))

    summaries = pd.concat(summaries)
    stats = pd.concat(stats)
    scaler = MinMaxScaler()
    selection = stats.groupby(['name', 'threshold', 'set']).agg(mean_purity=('mean_purity', 'mean'),
                                               mean_retention=('retention', 'mean'),
                                               med_purity=('med_purity','median'),
                                               min_purity=('mean_purity','min'),
                                                n_total_tcrs=('n_total_tcrs', 'sum'),
                                               n_singletons=('n_singletons','sum')).dropna()
    if len(selection)==0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        selection['scaled_mean_purity'] = scaler.fit_transform(selection['mean_purity'].values.reshape(-1, 1))
        selection['scaled_retention'] = scaler.fit_transform(selection['mean_retention'].values.reshape(-1, 1))
        selection['scaled_n_singletons'] = 1 - scaler.fit_transform(selection['n_singletons'].values.reshape(-1, 1))

        # selection['agg'] = selection.apply(
        #     lambda x: np.mean([x['scaled_mean_purity'], x['scaled_retention'], x['scaled_n_singletons']]), axis=1)
        # selection['hm_agg'] = selection.apply(
        #     lambda x: hmean([x['scaled_mean_purity'], x['scaled_retention'], x['scaled_n_singletons']]), axis=1)
        selection['agg3'] = selection.apply(
            lambda x: 0.3 * x['scaled_mean_purity'] + 0.4 * x['scaled_retention'] + 0.4 * x['scaled_n_singletons'], axis=1)
        t1 = selection.reset_index().query('set=="train"').sort_values('agg3', ascending=False).head(4).threshold.values
        t2 = selection.reset_index().query('set=="valid"').sort_values('agg3', ascending=False).head(4).threshold.values
        selected_ts = [x for x in t1 if x in t2]
        selection = selection.reset_index()
        for k, v in args.items():
            selection[k] = v
        selection = selection.query('threshold in @selected_ts').set_index(['name', 'threshold', 'set']+ list(args.keys()))
        return summaries.query('threshold in @selected_ts'), stats.query('threshold in @selected_ts'), selection

# In mainfolder containing all the triplet_tuning
# For subfolder in mainfolder
# get the args_fn from one of the fold folders
# args = read_args(args_fn)
mainfolder = '../output/triplet_tuning/'
subfolders = glob.glob(mainfolder + '*/')

train_df = pd.read_csv('../data/filtered/230927_nettcr_positives_only.csv')
dataset = FullTCRDataset(train_df, 0, 0, 22, 0, 0, 23, encoding='BL50LO', pad_scale=-20)
loader = dataset.get_dataloader(batch_size=1024, sampler=SequentialSampler)

summaries, stats, selected = [], [], []
for condition in tqdm(subfolders, desc='subfolders'):
    fold_dirs = sorted(glob.glob(condition + '*/'))
    args = read_args(glob.glob(fold_dirs[0] + '*args*.txt')[0])
    name = args['trp_type'] + '_' + str(args['margin']) + '_' + str(args['n_latent']) + '_' + str(args['w_kld']) + '_' + str(args['w_trp'])
    dist_type = 'l2' if args['trp_type'] == 'l2' else 'cos'
    fct = partial(wrapper, name=name, args=args, dist_type=dist_type,dataset=dataset,loader=loader)
    output = Parallel(n_jobs=5)(delayed(fct)(fdir=fd, fold=i) for (i, fd) in tqdm(enumerate(fold_dirs),desc='fold', leave=False))
    # for i, fd in tqdm(enumerate(fold_dirs), desc='fdir'):
    #     fct(fdir=fd, fold=i)

    summary = pd.concat([x[0] for x in output])
    stat = pd.concat([x[1] for x in output])
    select = pd.concat([x[2] for x in output])
    summaries.append(summary)
    stats.append(stat)
    selected.append(select)

summaries = pd.concat(summaries)
summaries.to_csv('../output/triplet_tuning/summaries.csv')
stats = pd.concat(stats)
stats.to_csv('../output/triplet_tuning/stats.csv')
selected = pd.concat(selected)
selected.to_csv('../output/triplet_tuning/selected.csv')