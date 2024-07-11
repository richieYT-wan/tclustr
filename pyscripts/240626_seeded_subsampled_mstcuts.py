import random
import networkx as nx
import glob

import numpy as np
import pandas as pd
import os, sys
import torch
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, make_filename
import argparse
from src.cluster_utils import resort_baseline, cluster_all_thresholds, load_model_full, get_latent_df, \
    get_distances_labels_from_latent
from src.networkx_utils import create_mst_from_distance_matrix, iterative_size_cut, iterative_topn_cut
from matplotlib import pyplot as plt
import seaborn as sns

def plot_size_vs_topn(score_size, purities_size, retentions_size, scores_topn, purities_topn, retentions_topn, outdir, filename):
    f, a = plt.subplots(1, 1, figsize=(7, 4))
    a2 = a.twinx()
    a.set_title(
        'No initial pruning ; Using Top-5 as initial cut ; distance-weighted betweenness\nTBCR comparison of size-cut and top-1 (silhouette) cut')

    a.plot(range(len(purities_topn)), purities_topn, lw=.75, ls=':', c='m',
           label='TBCR size(4)-cut avg purity')
    a.plot(range(len(retentions_topn)), retentions_topn, lw=.75, ls='-', c='m',
           label='TBCR size(4)-cut avg retention')
    a2.plot(range(len(scores_topn)), scores_topn, lw=.75, ls='--', c='m',
            label='TBCR size(4)-cut silhouette score')

    a.plot(range(len(purities_size)), purities_size, lw=1, ls=':', c='g',
           label='TBCR Top1-cut avg purity')
    a.plot(range(len(retentions_size)), retentions_size, lw=.75, ls='-', c='g',
           label='TBCR Top1-cut avg retention')
    a2.plot(range(len(score_size)), score_size, lw=.8, ls='--', c='g',
            label='TBCR Top1-cut silhouette score')

    # Align the tick marks
    a.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks from primary y-axis
    a2.yaxis.set_tick_params(which='both', length=0)  # Remove tick marks from secondary y-axis
    # Align the gridlines
    a.grid(True)
    a2.grid(False)  # Disable secondary y-axis gridlines
    a.set_ylabel('Retention // Purity (%)')
    a2.set_ylabel('Silhouette score')
    a2.set_ylim([-0.11, 0.16])
    a2.legend(bbox_to_anchor=(1.62, .25))
    a.set_xlabel('iteration')
    a.legend(bbox_to_anchor=(1.62, .88))
    f.savefig(f'{outdir}{filename}_silhouette_curves.png', dpi=150,
              bbox_inches='tight')

    pass


def create_tree_do_both_cuts(distance_matrix, args, index_col):
    G, tree, dist_matrix, values, labels, encoded_labels, label_encoder, raw_indices = create_mst_from_distance_matrix(distance_matrix, args['label_col'], index_col)
    _, _, clusters_size_cut, _, _, scores_size_cut, purities_size_cut, retentions_size_cut = iterative_size_cut(values, tree,
                                                                                                                args['initial_cut_threshold'],
                                                                                                                args['initial_cut_method'],
                                                                                                                # top_n = 1 here for the size cut
                                                                                                                args['top_n'], args['which_cut'], args['weighted'],
                                                                                                                args['verbose'], args['max_size'])
    _, _, clusters_topn_cut, _, _, scores_topn_cut, purities_topn_cut, retentions_topn_cut = iterative_topn_cut(values, tree,
                                                                                                                args['initial_cut_threshold'],
                                                                                                                args['initial_cut_method'],
                                                                                                                # top_n = 1 here for the topn cut ; score threshold=1
                                                                                                                args['top_n'], args['which_cut'], args['weighted'],
                                                                                                                args['verbose'], 1.0)
    # Getting the best scores and saving it
    best_score_size_index = np.argmax(scores_size_cut)
    best_score_topn_index = np.argmax(scores_topn_cut)
    best_silhouette_size = scores_size_cut[best_score_size_index]
    best_purity_size = purities_size_cut[best_score_size_index]
    best_retention_size = retentions_size_cut[best_score_size_index]
    best_silhouette_topn = scores_topn_cut[best_score_topn_index]
    best_purity_topn = purities_topn_cut[best_score_topn_index]
    best_retention_topn = retentions_topn_cut[best_score_topn_index]

    return pd.DataFrame([{'iteration':best_score_size_index , 'silhouette':best_silhouette_size, 'purity':best_purity_size, 'retention':best_retention_size},
                         {'iteration': best_score_topn_index, 'silhouette': best_silhouette_topn, 'purity': best_purity_topn, 'retention': best_retention_topn}],
                        index = ['size_cut', 'topn_cut']), # \
          # scores_size_cut, purities_size_cut, retentions_size_cut, \
          # scores_topn_cut, purities_topn_cut, retentions_topn_cut

def single_run_wrapper(latent_df, tbcralign, tcrdist, seed, index_col, args, n_jobs_clustering):
    random.seed(seed)
    subsample = []
    for p in sorted(latent_df.peptide.unique()):
        tmp = latent_df.query('peptide==@p')
        subsample.append(tmp.sample(min(len(tmp), random.randint(args['n_min'], args['n_max'])), random_state=seed))

    subsample = pd.concat(subsample)
    subsample_idxs = subsample[index_col].unique()
    dm_latent, values_latent, feats, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(subsample,
                                                                                                                    label_col=args['label_col'],
                                                                                                                    index_col=index_col)
    dm_tbcralign, values_tbcralign = resort_baseline(tbcralign, dm_latent, index_col=index_col)
    dm_tcrdist, values_tcrdist = resort_baseline(tcrdist, dm_latent, index_col=index_col)
    # Interval aggl clustering part
    tcrdist_clustering = cluster_all_thresholds(values_tcrdist, torch.rand([len(dm_tcrdist, 1)]), labels, encoded_labels, label_encoder, decimals=5, n_points=args['n_points'], n_jobs=n_jobs_clustering)
    tbcralign_clustering = cluster_all_thresholds(values_tbcralign, torch.rand([len(dm_tbcralign, 1)]), labels, encoded_labels, label_encoder, decimals=5, n_points=args['n_points'], n_jobs=n_jobs_clustering)
    latent_clustering = cluster_all_thresholds(values_latent, torch.rand([len(dm_latent, 1)]), labels, encoded_labels, label_encoder, decimals=5, n_points=args['n_points'], n_jobs=n_jobs_clustering)
    # MST cut clustering part
    stats_latent, scores_latent_size, purities_latent_size, retentions_latent_size, scores_latent_topn, purities_latent_topn, retentions_latent_topn = create_tree_do_both_cuts(dm_latent, args, index_col)
    stats_tbcralign, scores_tbcralign_size, purities_tbcralign_size, retentions_tbcralign_size, scores_tbcralign_topn, purities_tbcralign_topn, retentions_tbcralign_topn = create_tree_do_both_cuts(dm_tbcralign, args, index_col)
    stats_tcrdist, scores_tcrdist_size, purities_tcrdist_size, retentions_tcrdist_size, scores_tcrdist_topn, purities_tcrdist_topn, retentions_tcrdist_topn = create_tree_do_both_cuts(dm_tcrdist, args, index_col)
    results_df = pd.concat([stats_latent.assign(method='latent', seed=seed), stats_tbcralign.assign(method='tbcralign', seed=seed), stats_tcrdist.assign(method='tcrdist', seed=seed)])



def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
                        help='device to use for cuda')
    parser.add_argument('-f', '--file', dest='file', required=False, type=str,
                        default='../data/filtered/240326_nettcr_paired_NOswaps.csv',
                        help='filename of the input file')
    parser.add_argument('-tcrdist', '--tcrdist_file', dest='tcrdist_file', type=str,
                        default='../output/240411_ClusteringTests/dist_matrices/2404XX_OUTPUT_tbcralign_distmatrix_140peps_labeled.csv',
                        help='External labelled tcrdist baseline distance matrix')
    parser.add_argument('-tbcralign', '--tbcralign_file', dest='tbcralign_file', type=str,
                        default='../output/240411_ClusteringTests/dist_matrices/tcrdist3_distmatrix_140peps_new_labeled.csv',
                        help='External labelled tbcralign baseline distance matrix')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default=None, help='Additional output directory')
    parser.add_argument('-a1', '--a1_col', dest='a1_col', default='A1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a2', '--a2_col', dest='a2_col', default='A2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a3', '--a3_col', dest='a3_col', default='A3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b1', '--b1_col', dest='b1_col', default='B1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b2', '--b2_col', dest='b2_col', default='B2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b3', '--b3_col', dest='b3_col', default='B3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=8,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=6,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb3', '--max_len_b3', dest='max_len_b3', type=int, default=23,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlpep', '--max_len_pep', dest='max_len_pep', type=int, default=0,
                        help='Max seq length admitted for peptide. Set to 0 to disable adding peptide to the input')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool,
                        default=False,
                        help='Adding positional encoding to the sequence vector. False by default')
    """
    Models args 
    """
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory')
    parser.add_argument('-pt_file', type=str, required=False,
                        default='../output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj/epoch_4500_interval_checkpoint__kcv_fold_00_Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj.pt',
                        help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file', type=str, required=False,
                        default='../output/240618_NestedKCV_CNNVAE/Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj/checkpoint_best_kcv_fold_00_Nested_TwoStageCNNVAE_latent_128_kld_1e-2_ExpData_KFold_0_240618_1608_pDQhj_JSON_kwargs.json',
                        help='Path to the json file to reload the VAE model')
    parser.add_argument('-index_col', type=str, required=False, default='raw_index',
                        help='index col to sort both baselines and latent df')
    parser.add_argument('-label_col', type=str, required=False, default='peptide',
                        help='column containing the labels (eg peptide)')

"""
    Training hyperparameters & args
    """
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=300,
                        help='How many points to do for the bounds limits')
    parser.add_argument('-link', '--linkage', dest='linkage', type=str, default='complete',
                        help='Which linkage to use for AgglomerativeClustering')
    parser.add_argument('-icm', '--initial_cut_method', dest='initial_cut_method', default='top', type=str,
                        help='Initial cut method for the MST')
    parser.add_argument('-ict', '--initial_cut_threshold', dest='initial_cut_threshold', default=5,
                        help='Initial cut threshold for the MST. Must be int if initial_cut_method is "top", and float if initial_cut_method is "threshold"')
    parser.add_argument('-wc', '--which_cut', dest='which_cut', default='edge', type=str, help='"edge" or "node" cut')
    parser.add_argument('-wt', '--weighted', dest='weighted', default=True, type=str2bool, help='Weighted centrality cutting')
    parser.add_argument('-vb', '--verbose', dest='verbose', default=0, type=int, help='Verbosity level (0, 1, 2)')
    parser.add_argument('-ms', '--max_size', dest='max_size', default=4, type=int, help='max_size for size-cutting')
    parser.add_argument('-tn', '--top_n', dest='top_n', default=4, type=str, help='Top N edges to cut per iteration. (Default = 1)')
    """
    TODO: Misc. 
    """
    parser.add_argument('-kf', '--partition', dest='partition', type=int, default=1, help='which partition to filter the input_df')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-n_runs', '--n_runs', dest='n_runs', type=int, default=10000,
                        help='Number of runs for the bootstrap')
    parser.add_argument('-n_min', dest='n_min', type=int, default=50,
                        help='Minimum number of samples in subsampling step')
    parser.add_argument('-n_max', dest='n_max', type=int, default=90,
                        help='Max number of samples in subsampling step.')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=None,
                        help='Initial starting seed. Default = 13')
    parser.add_argument('-n_jobs', dest='n_jobs', default=8, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    print('Starting script')
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    assert not all([args[k] is None for k in ['model_folder', 'pt_file', 'json_file']]), \
        'Please provide either the path to the folder containing the .pt and .json or paths to each file (.pt/.json) separately!'

    # TODO : Remove this hardcoded behaviour and add to argument
    easy5peps = ['ELAGIGILTV', 'GILGFVFTL', 'LLWNGPMAV', 'RAKFKQLL', 'YLQPRTFLL']
    # Reading df and assert that index is present used to resort baselines
    df = pd.read_csv(args['file']).query(f'partition==@partition and peptide in @easy5peps')
    # Reading baselines
    tbcralign = pd.read_csv(args['tbcralign_file'], index_col=0).query('partition==@partition and partition in @easy5peps')
    tcrdist = pd.read_csv(args['tcrdist_file'], index_col=0).query('partition==@partition and partition in @easy5peps')

    # Indexing checks
    if args['index_col'] is not None:
        assert (args['index_col'] in df.columns), f'Provided index_col {args["index_col"]} not in df columns!'
        assert (args['index_col'] in tbcralign.columns), f'Provided index_col {args["index_col"]} not in tbcr columns!'
        assert (args['index_col'] in tcrdist.columns), f'Provided index_col {args["index_col"]} not in tcrdist columns!'
        index_col = args['index_col']
    else:
        assert 'raw_index' in df.columns or 'original_index' in df.columns, 'Index col not in df! (neither raw_index or original_index)'
        assert 'raw_index' in tcrdist.columns or 'original_index' in tcrdist.columns, 'Index col not in tcrdist df! (neither raw_index or original_index)'
        assert 'raw_index' in tbcralign.columns or 'original_index' in tbcralign.columns, 'Index col not in tbcralign df! (neither raw_index or original_index)'
        index_col = 'raw_index' if 'raw_index' in df.columns else 'original_index'

    # Output saving things
    unique_filename, kf, rid, connector = make_filename(args)
    outdir = '../output/'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")


    # Reading model and getting latent DF
    model = load_model_full(args['pt_file'], args['json_file'], map_location=args['device'])
    partition = args["partition"]
    # filtering one problematic index (nlvpmvatv - gilgfvftl clone)
    df = df.query(f'{index_col}!="VDJdb_4837"')
    latent_df = get_latent_df(model, df)

    # Subsampling bootstrapping
    if args['n_jobs']%2==0:
        if args['n_jobs']%4==0:
            n_jobs_clust = args['n_jobs']/4
            n_jobs_boots = args['n_jobs']/n_jobs_clust
        else:
            n_jobs_clust, n_jobs_boots = args['n_jobs']/2, args['n_jobs']/2
    else:
        n_jobs_clust = 1
        n_jobs_boots = args['n_jobs']

    for



