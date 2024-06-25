import glob
import os
import re
from functools import partial

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import auc, silhouette_score, adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score as ch_score, davies_bouldin_score as db_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm

from src.conv_models import CNNVAE
from src.datasets import FullTCRDataset
from src.models import TwoStageVAECLF, FullTCRVAE
from src.multimodal_datasets import MultimodalMarginalLatentDataset
from src.multimodal_models import BSSVAE, JMVAE
from src.multimodal_train_eval import predict_multimodal
from src.sim_utils import make_dist_matrix
from src.torch_utils import load_model_full
from src.train_eval import predict_model
from src.utils import get_palette, get_class_initcode_keys, mkdirs


##################################
#           PIPELINES            #
##################################


def do_baseline(baseline_distmatrix, identifier='baseline', label_col='peptide', n_points=500):
    cols = [str(x) for x in baseline_distmatrix.index.to_list()] if type(
        baseline_distmatrix.columns[0] == str) else baseline_distmatrix.index.to_list()
    dist_array = baseline_distmatrix[cols].values
    encoder = LabelEncoder()
    labels = baseline_distmatrix[label_col].values
    encoded_labels = encoder.fit_transform(labels)
    results = cluster_all_thresholds(dist_array, torch.rand([len(dist_array), 1]),
                                     labels, encoded_labels, encoder, n_points=n_points)
    results['input_type'] = identifier
    return results


def sort_key(item):
    if item == 'last_epoch':
        return float('inf'), 1  # last_epoch will be the second last
    elif item == 'checkpoint_best':
        return float('inf'), 2  # checkpoint_best will be last
    else:
        # Extract the number from strings like 'epoch_6000' and convert to integer
        return int(re.search(r'\d+', item).group()), 0


def run_interval_clustering(model_folder, input_df, index_col, identifier='VAEmodel', n_points=1500, n_jobs=1):
    # Assuming we are using VAE based models
    if not model_folder.endswith("/"):
        model_folder = model_folder + '/'
    files = glob.glob(f'{model_folder}*.pt')
    js = glob.glob(f'{model_folder}*JSON*.json')[0]
    # Get the epochs and sort them
    epochs = ['_'.join(os.path.basename(x.replace(model_folder, '')).split('_')[:2]) for x in files]
    sorted_epochs = sorted(epochs, key=sort_key)
    # Get the mapping to load the files in the correct order
    map_epochs_files = {k: v for k, v in zip(epochs, files)}

    # Run the analysis for each of these VAE models
    cat_results = []
    best_dm = None
    best_auc = -1
    for name in sorted_epochs:
        if 'last_epoch' in name:
            continue
        checkpoint = map_epochs_files[name]
        model, model_json = load_model_full(checkpoint, js, map_location='cpu', return_json=True)
        if hasattr(model, 'vae'):
            model = model.vae
        latent_df = get_latent_df(model, input_df)
        dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
            latent_df, index_col=index_col)

        results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points, n_jobs=n_jobs)
        results['input_type'] = f'{identifier}_{name}'
        retentions = results['retention'].values[1:-1]
        purities = results['mean_purity'].values[1:-1]
        # Small fix to set a max threshold for retention at 0.98 to avoid cases where 
        # we have 100% retention with one large garbage cluster very unpure and 2-3 small clusters 100% purity
        # that would give an overall mean_purity of 70% or somethings
        p70_r35_auc = get_retpur_auc(retentions, purities, min_retention=0.35, min_purity=0.70, max_retention=0.975)

        # Saving just the best in case nothing beats the best AUC and no best_dm has been found
        if 'checkpoint_best' in checkpoint and best_dm is None and 'interval' not in checkpoint:
            best_dm = dist_matrix
            best_auc = p70_r35_auc
            best_name = 'checkpoint_best'
        # Then do the actual auc check
        if p70_r35_auc > best_auc:
            best_dm = dist_matrix
            best_auc = p70_r35_auc
            best_name = name
        cat_results.append(results)
    cat_results = pd.concat(cat_results)
    return cat_results, best_dm, best_name


def plot_retention_purities(runs, title=None, fn=None, palette='tab10', add_clustersize=False,
                       figsize=(9, 9), legend=True, outdir=None):
    # plotting options
    sns.set_palette(palette, n_colors=len(runs.input_type.unique()))
    f, a = plt.subplots(1, 1, figsize=figsize)
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Mean Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    if add_clustersize:
        ax2 = a.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_yscale('log', base=2)
        ax2.set_ylabel('Mean Cluster Size (Log2)', fontweight='semibold', fontsize=14)
    for input_type in runs.input_type.unique():
        query = runs.query('input_type==@input_type')
        retentions = query['retention'][1:-1].values
        purities = query['mean_purity'][1:-1].values
        if add_clustersize:
            cluster_sizes = query['mean_cluster_size'].values[1:-1]

        if input_type == "TBCRalign":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls=':', c='g', lw=1)
            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='x', lw=0.25, s=6, c='g')

        elif input_type == "tcrdist3":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls=':', c='m', lw=1)
            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='.', lw=0.25, s=6, c='m')

        else:
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='--', lw=1.1)
            if add_clustersize:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='*', lw=0.25, s=6)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')
    if legend:
        a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
                 prop={'weight': 'semibold', 'size': 12}, loc='lower left')
    f.suptitle(f'{title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fn is not None:
        if outdir is not None:
            mkdirs(outdir)
            fn = f'{outdir}{fn}'
        f.savefig(f'{fn}.png', dpi=200)
    return f,a

def run_interval_plot_pipeline(model_folder, input_df, index_col, label_col, tbcr_dm, identifier='', n_points=250,
                               baselines=None, plot_title='None', fig_fn=None, n_jobs=1):
    try:
        interval_runs, best_dm, best_name = run_interval_clustering(model_folder, input_df, index_col, identifier, n_points, n_jobs=n_jobs)
    except:
        print('\n\n\n\n', model_folder, identifier, '\n\n\n\n')
        raise ValueError(model_folder, identifier)
    # Only do TBCRalign and tcrdist3 because we know we are better than KernelSim by a good margin
    tbcr_dm, tbcr_da = resort_baseline(tbcr_dm, best_dm, index_col)
    agg_results = do_agg_clustering_best(best_dm, tbcr_dm, index_col, label_col, n_points)
    interval_runs = pd.concat([baselines.query('input_type=="TBCRalign"'),
                               baselines.query('input_type == "tcrdist3"'),
                               interval_runs,
                               agg_results.assign(input_type=f'agg_{best_name}')])
    interval_runs['input_type'] = interval_runs['input_type'].apply(lambda x: x.replace(identifier,'').lstrip('_').rstrip('_'))
    # plotting options
    sns.set_palette('gnuplot2', n_colors=len(interval_runs.input_type.unique()) - 2)
    f, a = plt.subplots(1, 1, figsize=(15, 15))
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Avg Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    for input_type in interval_runs.input_type.unique():
        query = interval_runs.query('input_type==@input_type')
        retentions = query['retention'][1:-1].values
        purities = query['mean_purity'][1:-1].values
        if input_type == "TBCRalign":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='g', lw=1.)
        elif input_type == "tcrdist3":
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='y', lw=1.)
        else:
            a.plot(retentions, purities, label=input_type.lstrip('_'), ls='--', lw=1.)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')

    a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
             prop={'weight': 'semibold', 'size': 12})
    f.suptitle(f'{plot_title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fig_fn is not None:
        f.savefig(f'{fig_fn}.png', dpi=200)
    return interval_runs


def do_agg_clustering_best(input_dm, other_dm_labelled, index_col, label_col='peptide', n_points=500):
    # provide either a model path or an actual loaded model
    baseline_dm, baseline_da = resort_baseline(other_dm_labelled, input_dm, index_col)
    input_da = input_dm.iloc[:len(baseline_da), :len(baseline_da)].values
    labels = input_dm[label_col].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit(labels)
    agg_array = 1 - np.multiply(1 - input_da, 1 - baseline_da)
    results = cluster_all_thresholds(agg_array, torch.rand([len(agg_array), 1]),
                                     labels, encoded_labels, label_encoder, n_points=n_points)
    return results


def do_agg_clustering(model, input_df, other_dm_labelled, n_points=500):
    # provide either a model path or an actual loaded model
    if type(model) == str:
        model = get_model(model)
    # Assuming both input_df and other_dm_labelled have the raw_index...
    latent_df = get_latent_df(model, input_df)
    # Sort it accordingly
    latent_df = latent_df.set_index('raw_index').loc[other_dm_labelled['raw_index'].values].reset_index()
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    other_array = other_dm_labelled.iloc[:other_dm_labelled, :other_dm_labelled].values
    agg_array = 1 - np.multiply((1 - dist_array, 1 - other_array))
    results = cluster_all_thresholds(agg_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    return results


def pipeline_best_model_clustering(model_folder, input_df, name, n_points=500):
    # Assuming we are using VAE based models
    model = get_model(model_folder,
                      map_location='cpu')
    latent_df = get_latent_df(model, input_df)
    print(len(latent_df))
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    results['input_type'] = name
    return results


def cluster_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, threshold):
    c = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=threshold, linkage='complete')
    c.fit(dist_array)
    return get_all_metrics(threshold, features, c, dist_array, labels, encoded_labels, label_encoder)


# Here, do a run ith only the best
def cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder,
                           decimals=5, n_points=500, n_jobs=1):
    # Getting clustering at all thresholds
    limits = get_linspace(dist_array, decimals, n_points)
    if n_jobs>1:
        print('HERE')
        wrapper = partial(cluster_single_threshold, dist_array=dist_array, features=features, labels=labels, encoded_labels=encoded_labels, label_encoder=label_encoder)
        results = Parallel(n_jobs=n_jobs)(delayed(wrapper)(threshold=t) for t in tqdm(limits))
    else:
        results = []
        for t in tqdm(limits):
            results.append(cluster_single_threshold(dist_array, features, labels, encoded_labels, label_encoder, t))
    results = pd.DataFrame(results).sort_values('threshold')
    results['retention'] = (dist_array.shape[0] - results['n_singletons']) / dist_array.shape[0]
    return results


def vae_clustering_pipeline(model_folder, input_df, name, dataset_params=None, n_points=500):
    model = get_model(model_folder, map_location='cpu')
    latent_df = get_latent_df(model, input_df, dataset_params)
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(
        latent_df)
    results = cluster_all_thresholds(dist_array, features, labels, encoded_labels, label_encoder, n_points=n_points)
    results['input_type'] = name
    return results


# Here, do a run ith only the best
def plot_pipeline(results, b, plot_title='None', fig_fn=None, filter=None, palette=None, more=False,
                  add_cluster_size=False):
    runs = pd.concat([b, results])
    # plotting options
    if filter is None:
        filter = ['TBCRalign', 'KernelSim', 'tcrdist3'] + list(results.input_type.unique())

    if palette is None:
        palette = 'gnuplot2'
    if more:
        palette = get_palette(palette, n_colors=len(filter) - 3)
    else:
        palette = sns.color_palette(palette, n_colors=len(filter) - 3)

    sns.set_palette(palette)
    f, a = plt.subplots(1, 1, figsize=(9, 9))
    a.set_xlim([0, 1])
    a.set_ylim([0, 1])
    a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
    a.set_ylabel('Avg Purity', fontweight='semibold', fontsize=14)
    # Setting major ticks
    major_ticks = np.arange(0, 1.1, 0.1)
    a.set_xticks(major_ticks)
    a.set_yticks(major_ticks)
    # Setting minor ticks
    minor_ticks = np.arange(0, 1.1, 0.05)
    a.set_xticks(minor_ticks, minor=True)
    a.set_yticks(minor_ticks, minor=True)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    print(runs.duplicated().any())
    if add_cluster_size:
        ax2 = a.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_yscale('log', base=2)

    for i, input_type in enumerate(filter):
        query = runs.query('input_type==@input_type')
        retentions = query['retention'].values[1:-1]
        purities = query['mean_purity'].values[1:-1]
        print(input_type, '\t', round(get_retpur_auc(retentions, purities), 4))
        ls = '-' if i % 2 == 0 else '--'
        if add_cluster_size:
            cluster_sizes = query['mean_cluster_size'].values[1:-1]
        # Plotting baselines with fixed styles colors etc
        if input_type == "TBCRalign":
            a.plot(retentions, purities, label='TBCRalign', ls=':', c='k', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='x', lw=0.5, s=8, c='k')
        elif input_type == "KernelSim":
            a.plot(retentions, purities, label='KernelSim', ls=':', c='m', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='v', lw=0.1, s=8, c='m')
        elif input_type == "tcrdist3":
            a.plot(retentions, purities, label='tcrdist3', ls=':', c='y', lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='*', lw=0.1, s=8, c='y')
        # Plotting the actual things
        else:
            a.plot(retentions, purities, label=input_type.lstrip('_').replace('_', ' ').replace('checkpoint best', ''),
                   ls=ls, lw=1.)
            if add_cluster_size:
                ax2.scatter(retentions, cluster_sizes, label=input_type.lstrip('_'), marker='+', lw=1.15, s=12)

    a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
    a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
    a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')

    a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
             prop={'weight': 'semibold', 'size': 12})

    f.suptitle(f'{plot_title}', fontweight='semibold', fontsize=15)
    f.tight_layout()
    if fig_fn is not None:
        f.savefig(f'../output/240411_ClusteringTests/{fig_fn}.png', dpi=200)
    return runs


##################################
#      LOAD MODEL / LATENT       #
##################################

def resort_baseline(baseline_dm, input_dm, index_col,
                    cols=('peptide', 'original_peptide', 'binder', 'partition')):
    """
    Resorts the baseline_dm to match input_dm in order to do agg_clustering
    Args:
        baseline_dm:
        input_dm:
        index_col:

    Returns:

    """
    cols = list(cols)
    if index_col not in cols:
        cols.append(index_col)
    baseline_copy = baseline_dm.copy()
    baseline_copy = baseline_copy.drop_duplicates(index_col)
    original_index_col = baseline_dm.index.name
    reindex = input_dm[index_col].values
    baseline_copy = baseline_copy.reset_index().set_index(index_col).loc[reindex].reset_index().set_index(
        original_index_col)
    idxcols = [str(x) for x in baseline_copy.index.to_list()] if type(
        [x for x in baseline_copy.columns if x not in cols][0]) == str else baseline_copy.index.to_list()
    # print(idxcols+cols)
    baseline_copy = baseline_copy[idxcols + cols]
    # baseline_copy = baseline_copy.reset_index().set_index(original_index_col)
    baseline_array = baseline_copy.iloc[:len(baseline_copy), :len(baseline_copy)].values
    return baseline_copy, baseline_array


def get_retpur_auc(retentions, purities, min_retention=0.0, max_retention=1.0, min_purity=0.0, max_purity=1.0):
    ridx = np.where((min_retention <= retentions) & (retentions <= max_retention))
    pidx = np.where((min_purity <= purities) & (purities <= max_purity))
    idx = np.intersect1d(ridx, pidx)
    if len(idx) <= 1:
        return -1
    retentions = retentions[idx]
    purities = purities[idx]
    return auc(retentions, purities)


def get_all_rpauc(retentions, purities, input_type=None, name=None):
    total_auc = get_retpur_auc(retentions, purities)
    p60_auc = get_retpur_auc(retentions, purities, min_purity=0.6)
    p70_auc = get_retpur_auc(retentions, purities, min_purity=0.7)

    p60_r40_auc = get_retpur_auc(retentions, purities, min_retention=0.4, min_purity=0.6, max_retention=0.975)
    p70_r35_auc = get_retpur_auc(retentions, purities, min_retention=0.35, min_purity=0.70, max_retention=0.975)
    p70_r50_auc = get_retpur_auc(retentions, purities, min_retention=0.5, min_purity=0.70, max_retention=0.975)
    out = {'input_type': input_type,
           'name': name,
           'total_auc': total_auc,
           'p70_auc': p70_auc,
           'p60_auc': p60_auc,
           'p60_r40_auc': p60_r40_auc,
           'p70_r35_auc': p70_r35_auc,
           'p70_r50_auc': p70_r50_auc}

    return out

def get_all_inputs_rpauc(df, input_col ='input_type'):
    # Utility function to return the rpauc df for all groups and sort by best
    return pd.json_normalize(df.groupby('input_type').apply(lambda group: get_all_rpauc(group['retention'].values[1:-1], 
                                                              group['mean_purity'].values[1:-1],
                                                              group[input_col].iloc[0]))).drop(columns=['name']).set_index(input_col).sort_values('p70_r35_auc',ascending=False)




def get_model(folder, map_location='cpu'):
    pt = glob.glob(folder + '/*checkpoint_best*.pt')
    pt = [x for x in pt if 'interval' not in x][0]
    js = glob.glob(folder + '/*checkpoint*.json')[0]
    model = load_model_full(pt, js, map_location='cpu')
    # Extract the vae part if the model comes from a two stage VAE
    if type(model) == TwoStageVAECLF:
        model = model.vae
    model.eval()
    return model


def get_latent_df(model, df, dataset_params: dict = None):
    # Init dataset and pred fct depending on model type
    dataset_params = dict(max_len_a1=7, max_len_a2=8, max_len_a3=22,
                          max_len_b1=6, max_len_b2=7, max_len_b3=23, max_len_pep=0,
                          encoding='BL50LO', pad_scale=-20,
                          a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                          pep_col='peptide') if dataset_params is None else dataset_params

    if hasattr(model, 'vae'):
        model = model.vae

    if model.max_len > 7 + 8 + 22 + 6 + 7 + 23:
        dataset_params['max_len_pep'] = 12

    elif model.max_len <= 22 + 23:
        dataset_params['max_len_a1'] = 0
        dataset_params['max_len_a2'] = 0
        dataset_params['max_len_b1'] = 0
        dataset_params['max_len_b2'] = 0
        dataset_params['max_len_pep'] = 0

    dataset_params['add_positional_encoding'] = model.add_positional_encoding

    if type(model) == FullTCRVAE:
        print(dataset_params)
        dataset = FullTCRDataset(df, **dataset_params)
        dataloader = dataset.get_dataloader(512, SequentialSampler)
        latent_df = predict_model(model, dataset, dataloader)

    elif type(model) == CNNVAE:
        dataset_params['conv'] = True
        dataset = FullTCRDataset(df, **dataset_params)
        dataloader = dataset.get_dataloader(512, SequentialSampler)
        latent_df = predict_model(model, dataset, dataloader)

    # TODO: This part probly not properly handled
    elif type(model) in [BSSVAE, JMVAE]:
        dataset_params['pair_only'] = True
        dataset_params['return_pair'] = type(model) == JMVAE
        dataset_params['modality'] = 'tcr'
        dataset = MultimodalMarginalLatentDataset(model, df, **dataset_params)
        latent_df = df.copy()
        zdim = dataset.z.shape[1]
        latent_df[[f'z_{i}' for i in range(zdim)]] = dataset.z

    return latent_df


def get_distances_labels_from_latent(latent_df, label_col='peptide', seq_cols=('A1', 'A2', 'A3', 'B1', 'B2', 'B3'),
                                     index_col='original_index'):
    # Columns for making distmatrix
    rest_cols = list(
        x for x in latent_df.columns if x in ['peptide', 'original_peptide', 'origin', 'binder', index_col])
    # Getting distmatrix and arrays
    dist_matrix = make_dist_matrix(latent_df, label_col, seq_cols, cols=rest_cols)
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values
    # Getting label encoder and features for computing metrics
    features = latent_df[[z for z in latent_df.columns if z.startswith('z_')]].values
    label_encoder = LabelEncoder()
    labels = dist_matrix[label_col].values
    encoded_labels = label_encoder.fit_transform(labels)
    return dist_matrix, dist_array, features, labels, encoded_labels, label_encoder


##################################
#        tbcralign fcts          #
##################################
def get_merged_distances_labels(dist_matrix, original_df, index_tcr_df, label_col='peptide', query_subset=None):
    # Assumes a square matrix with no other columns, and that the original_df and index_tcr_df match
    merged = pd.merge(index_tcr_df, original_df[
        [x for x in original_df.columns if x in ['seq_id', 'peptide', 'partition', 'binder', 'origin', 'fulltcr']]],
                      left_on=['q_index', 'tcr'], right_on=['seq_id', 'fulltcr'])

    assert ((merged['seq_id'] == merged['q_index']).all() and (merged['tcr'] == merged['fulltcr']).all()), 'fuck'
    merged = merged.set_index('q_index')[
        [x for x in merged.columns if x in ['peptide', 'partition', 'binder', 'origin']]]
    merged_dist_matrix = pd.merge(dist_matrix, merged, left_index=True, right_index=True)
    extra_cols = merged_dist_matrix.columns.difference(dist_matrix.columns)

    if query_subset is not None:
        query = merged_dist_matrix.query(query_subset)
        merged_dist_matrix = query[list(str(x) for x in query.index) + list(extra_cols)]

    return merged_dist_matrix, extra_cols


def get_distances_labels_from_distmatrix(dist_matrix, original_df, index_tcr_df, label_col='peptide',
                                         query_subset=None):
    merged_dist_matrix, extra_cols = get_merged_distances_labels(dist_matrix, original_df, index_tcr_df, label_col,
                                                                 query_subset)
    dist_array = merged_dist_matrix.iloc[:, :-len(extra_cols)].values
    features = torch.randn([dist_array.shape[0], 3])
    label_encoder = LabelEncoder()
    labels = merged_dist_matrix[label_col].values
    encoded_labels = label_encoder.fit_transform(labels)
    return merged_dist_matrix, dist_array, features, labels, encoded_labels, label_encoder


##################################
#            METRICS             #
##################################

def get_purity(counts):
    # Purity in absolute % of a cluster, taking the majority label
    # high = better
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_counts[list(sorted_counts.keys())[0]] / sum(sorted_counts.values())


def get_mixity(counts):
    # how many different labels are inside a cluster, weighted by the number of members
    # low = better
    return len(counts.keys()) / sum(counts.values())


def get_coherence(dist_array):
    # Assumes dist_array is the subset of the distance array for a given cluster label
    # mean distance within a cluster
    # low = better

    # get upper triangle mask without the diagonale
    mask = np.triu(np.ones(dist_array.shape), k=0) - np.eye(dist_array.shape[0])
    flat_array = dist_array[mask == 1]
    return np.mean(flat_array)


def get_purity_mixity_coherence(cluster_label: int,
                                true_labels: list,
                                pred_labels: list,
                                dist_array: np.array,
                                label_encoder):
    """
        For a given cluster label (int) returned by clustering.labels_,
        Return the purity, mixity, coherence, cluster_size, and scale (==cluster_size/total_size)
        scale should be used to get a weighted average metric at the end
    """
    indices = np.where(pred_labels == cluster_label)[0]
    cluster_size = len(indices)
    if cluster_size <= 1:
        # return np.nan, np.nan, np.nan, 1, 1/len(true_labels)
        return {'purity': np.nan, 'coherence': np.nan, 'cluster_size': 1}

    # Query the subset of the true labels belonging to this cluster using indices
    # Convert to int label encodings in order to use np.bincount to get purity and mixity
    subset = true_labels[indices]
    encoded_subset = label_encoder.transform(subset)
    counts = {i: k for i, k in enumerate(np.bincount(encoded_subset)) if k > 0}
    purity = get_purity(counts)
    # mixity = get_mixity(counts)
    # index the distance matrix and return the mean distance within this cluster (i.e. coherence)
    coherence = get_coherence(dist_array[indices][:, indices])

    # return purity, mixity, coherence, cluster_size, cluster_size / len(true_labels)
    return {'purity': purity, 'coherence': coherence, 'cluster_size': cluster_size}


def get_all_metrics(t, features, c, array, true_labels, encoded_labels, label_encoder):
    n_cluster = np.sum((np.bincount(c.labels_) > 1))
    n_singletons = (np.bincount(c.labels_) == 1).sum()
    try:
        s_score = silhouette_score(features, c.labels_, metric='cosine')
    except:
        s_score = np.nan
    try:
        c_score = ch_score(features, c.labels_)
    except:
        c_score = np.nan
    try:
        d_score = db_score(features, c.labels_)
    except:
        d_score = np.nan
    try:
        ari_score = adjusted_rand_score(encoded_labels, c.labels_)
    except:
        ari_score = np.nan

    xd = pd.concat(
        [pd.DataFrame(get_purity_mixity_coherence(k, true_labels, c.labels_, array, label_encoder), index=[0])
         for k in set(c.labels_)]).dropna()
    mean_purity = xd['purity'].mean()
    mean_coherence = xd['coherence'].mean()
    mean_cs = xd['cluster_size'].mean()
    nc_07 = len(xd.query('purity>=0.7'))
    return {'threshold': t,
            'n_cluster': n_cluster, 'n_singletons': n_singletons,
            'n_cluster_over_70p': nc_07,
            'mean_purity': xd['purity'].mean(),
            'min_purity': xd['purity'].min(),
            'max_purity': xd['purity'].max(),
            'mean_coherence': xd['coherence'].mean(),
            'min_coherence': xd['coherence'].min(),
            'max_coherence': xd['coherence'].max(),
            'mean_cluster_size': xd['cluster_size'].mean(),
            'min_cluster_size': xd['cluster_size'].min(),
            'max_cluster_size': xd['cluster_size'].max(),
            'silhouette': s_score,
            'ch_index': c_score, 'db_index': d_score, 'ARI': ari_score}


def get_bounds(array, decimals=5):
    lower_bound = array[array > 0].min()
    upper_bound = array.max()
    factor = 10 ** decimals
    return np.floor(lower_bound * factor) / factor, np.ceil(upper_bound * factor) / factor


def get_linspace(array, decimals=5, n_points=1500):
    return np.round(np.linspace(*get_bounds(array, decimals), n_points), decimals)
