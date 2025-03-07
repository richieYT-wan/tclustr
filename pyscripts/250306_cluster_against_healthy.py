import argparse
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from src.cluster_utils import *
from src.networkx_utils import *
from src.torch_utils import load_model_full
from src.utils import str2bool, make_filename, pkl_dump
from datetime import datetime as dt
from scipy.cluster.hierarchy import linkage, leaves_list


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default='cpu', type=str,
                        help='device to use for cuda')
    parser.add_argument('-hf', '--healthy_file', dest='healthy_file', type=str,
                        default='../data/OTS/garner_merged/garner_merged_41_42_43.csv')
    parser.add_argument('-if', '--file', dest='file', type=str,
                        default='../data/IMMREP25/test.csv_fmt4TCRcluster_uniq',
                        help='filename of the input file')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default=None, help='Additional output directory')
    """
        Sampling args
    """

    parser.add_argument('-ratio', dest='ratio', type=float, default=1,
                        help='ratio of healthy to sample')
    parser.add_argument('-lc', '--low_count', type=str2bool, default=True,
                        help='Low Count for sampling healthy')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=0,
                        help='Seed for sampling')
    parser.add_argument('-reidx_h', '--reindex_healthy', type=str2bool, default=False,
                        help='re-assign index to sampled healthy')
    parser.add_argument('-low_memory', type=str2bool, default=False,
                        help='whether to use "low memory merge mode. Might get wrong results...')
    """
    Training hyperparameters & args
    """

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-mp', '--min_purity', dest='min_purity', type=float, default=.8)
    parser.add_argument('-ms', '--min_size', dest='min_size', type=int, default=5)
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=500,
                        help='How many points to do for the bounds limits')
    parser.add_argument('-link', '--linkage', dest='linkage', type=str, default='complete',
                        help='Which linkage to use for AgglomerativeClustering')
    parser.add_argument('-s_agg', '--silhouette_aggregation', default='micro', dest='silhouette_aggregation',
                        help='whether to use micro or macro for the silhouette score aggregation')
    parser.add_argument('--save_dm', dest='save_dm', default=False, type=str2bool)
    """
    TODO: Misc. 
    """
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-n_jobs', dest='n_jobs', default=8, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def get_linkage_sorted_dm(dist_matrix, method='complete', metric='cosine', optimal_ordering=True):
    rest_cols = [x for x in dist_matrix.columns if x not in dist_matrix.index]
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values
    linkage_matrix = linkage(dist_array, method=method, metric=metric,
                             optimal_ordering=optimal_ordering)  # Use your preferred method (e.g., 'ward', 'average')
    dendrogram_order = leaves_list(linkage_matrix)  # Get the order of rows/columns
    sorted_dm = dist_matrix.iloc[dendrogram_order]
    sorted_da = dist_matrix.iloc[dendrogram_order, dendrogram_order].values
    sorted_dm = sorted_dm[sorted_dm.index.to_list() + rest_cols]
    return sorted_dm, sorted_da


def pipeline_fct(latent_df, label_col, seq_cols, index_col, rest_cols,
                 outdir, name,
                 args):
    dist_matrix, dist_array, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(latent_df,
                                                                                                         label_col,
                                                                                                         seq_cols,
                                                                                                         index_col,
                                                                                                         rest_cols,
                                                                                                         args[
                                                                                                             'low_memory'])
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values
    # print('\nOptim\n')
    if args['debug']:
        args['n_points']=15
        args['n_jobs']=-1
    optimisation_results = agglo_all_thresholds(dist_array, dist_array, labels, encoded_labels, label_encoder, 5,
                                                args['n_points'], args['min_purity'], args['min_size'], 'micro',
                                                args['n_jobs'])
    # print('Got optim')
    optimisation_results['best'] = False
    optimisation_results.loc[
        optimisation_results.iloc[:int(0.8 * len(optimisation_results))]['silhouette'].idxmax(), 'best'] = True
    plot_sprm(optimisation_results, fn=f'{outdir}{name}optimisation_curves')
    threshold = optimisation_results.query('best')['threshold'].item()
    optimisation_results[['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']] = optimisation_results[
        ['silhouette', 'mean_purity', 'retention', 'mean_cluster_size']].round(3)
    optimisation_results['max_cluster_size'] = optimisation_results['max_cluster_size'].round(0)
    optimisation_results.to_csv(f'{outdir}{name}optimisation_results_df.csv')
    # print('saved optim')

    metrics, clusters_df, c = agglo_single_threshold(dist_array, dist_array, labels, encoded_labels,
                                                     label_encoder, threshold,
                                                     min_purity=args['min_purity'], min_size=args['min_size'],
                                                     silhouette_aggregation='micro',
                                                     return_df_and_c=True)
    dist_matrix['cluster_label'] = c.labels_
    keep_columns = ['index_col', 'cluster_label']
    results_df = pd.merge(latent_df, dist_matrix[keep_columns], left_on=index_col, right_on=index_col)
    # print('Merged dfs')
    clusters_df.to_csv(f'{outdir}{name}clusters_summary.csv', index=False)
    # Here now sort DF / results + plot heatmap
    sorted_dm, sorted_da = get_linkage_sorted_dm(dist_matrix, 'complete', 'cosine', True)
    if args['save_dm']:
        sorted_dm.to_csv(f'{outdir}{name}sorted_cosine_distance_matrix.csv')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    sns.heatmap(sorted_da, ax=ax, square=True, cmap='viridis', xticklabels=False, yticklabels=False)
    results_df = results_df.set_index(index_col).loc[sorted_dm[index_col]].reset_index()
    fig.savefig(f'{outdir}{name}complete_cosine_sorted_heatmap.png', dpi=150)
    results_df.to_csv(f'{outdir}{name}TCRcluster_results.csv', index=False)


def main():
    start = dt.now()
    print('Starting MST-cut pyscript')
    sns.set_style('darkgrid')
    args = vars(args_parser())
    unique_filename, kf, rid, connector = make_filename(args)
    unique_filename = f'seed_{args["seed"]:04}_ratio_{args["ratio"]}_'+unique_filename
    outdir = '../output/'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)
    # dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    # read input df and prep
    df = pd.read_csv(args['file'])
    df['label'] = 'sample'
    df['index_col'] = [f'sample_{i:05}' for i in range(len(df))]
    # Read background df and sample
    healthy_df = pd.read_csv(args['healthy_file'])
    if args['low_count']:
        healthy_df = healthy_df.query('mean_count==1')
    background = healthy_df.sample(n=int(len(df) * args['ratio']), random_state=args['seed'])
    background['label'] = 'background'
    if 'index_col' not in background.columns or args['reindex_healthy']:
        background['index_col'] = [f'background_{i:05}' for i in range(len(background))]

    merged_df = pd.concat([df, background])

    model_paths = {'OSNOTRP': {'pt': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni.pt',
                               'json': '../models/OneStage_NoTriplet_6omni/checkpoint_best_OneStage_NoTriplet_6omni_JSON_kwargs.json'},
                   'OSCSTRP': {'pt': '../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ.pt',
                               'json': '../models/OneStage_CosTriplet_ER8wJ/checkpoint_best_OneStage_CosTriplet_ER8wJ_JSON_kwargs.json'},
                   'TSNOTRP': {
                       'pt': '../models/TwoStage_NoTriplet_N1jMC/epoch_4500_interval_checkpoint_TwoStage_NoTriplet_N1jMC.pt',
                       'json': '../models/TwoStage_NoTriplet_N1jMC/checkpoint_best_TwoStage_NoTriplet_N1jMC_JSON_kwargs.json'},
                   'TSCSTRP': {
                       'pt': '../models/TwoStage_CosTriplet_jyGpd/epoch_4500_interval_checkpoint_TwoStage_CosTriplet_jyGpd.pt',
                       'json': '../models/TwoStage_CosTriplet_jyGpd/checkpoint_best_TwoStage_CosTriplet_jyGpd_JSON_kwargs.json'}}

    model_os_notrp = load_model_full(model_paths['OSNOTRP']['pt'], model_paths['OSNOTRP']['json'],
                                     map_location=args['device'], verbose=True)
    if not args['debug']:
        model_ts_notrp = load_model_full(model_paths['TSNOTRP']['pt'], model_paths['TSNOTRP']['json'],
                                         map_location=args['device'], verbose=True)
        model_os_cstrp = load_model_full(model_paths['OSCSTRP']['pt'], model_paths['OSCSTRP']['json'],
                                         map_location=args['device'], verbose=True)
        model_ts_cstrp = load_model_full(model_paths['TSCSTRP']['pt'], model_paths['TSCSTRP']['json'],
                                         map_location=args['device'], verbose=True)

    rest_cols = [x for x in merged_df.columns if x not in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']]
    seq_cols = ('A1', 'A2', 'A3', 'B1', 'B2', 'B3')
    latent_df_os_notrp = get_latent_df(model_os_notrp, merged_df)
    if not args['debug']:
        latent_df_ts_notrp = get_latent_df(model_ts_notrp, merged_df)
        latent_df_os_cstrp = get_latent_df(model_os_cstrp, merged_df)
        latent_df_ts_cstrp = get_latent_df(model_ts_cstrp, merged_df)

    pipeline_fct(latent_df_os_notrp, 'label', seq_cols, 'index_col', rest_cols,
                 outdir, f'{unique_filename}OS_NOTRP_', args)
    if not args['debug']:
        pipeline_fct(latent_df_os_cstrp, 'label', seq_cols, 'index_col', rest_cols,
                     outdir, f'{unique_filename}OS_CSTRP_', args)
        pipeline_fct(latent_df_ts_notrp, 'label', seq_cols, 'index_col', rest_cols,
                     outdir, f'{unique_filename}TS_NOTRP_', args)
        pipeline_fct(latent_df_ts_cstrp, 'label', seq_cols, 'index_col', rest_cols,
                     outdir, f'{unique_filename}TS_CSTRP_', args)
    # if args['save_sample']:
    merged_df.to_csv(f'{outdir}{unique_filename}_saved_merged_sample_df.csv', index=False)


if __name__=='__main__':
    main()