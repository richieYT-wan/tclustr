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


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default='cpu', type=str,
                        help='device to use for cuda')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input file')
    parser.add_argument('-tcrdist', '--tcrdist_file', dest='tcrdist_file', type=str,
                        default=None, help='External labelled tcrdist baseline')
    parser.add_argument('-tbcralign', '--tbcralign_file', dest='tbcralign_file', type=str,
                        default=None, help='External labelled tbcralign baseline')
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
    parser.add_argument('-model', type=str, default='TSCSTRP',
                        help='which model to use ; can be "OSNOTRP", "OSCSTRP", "TSNOTRP", "TSCSTRP"')
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory')
    parser.add_argument('-pt_file_os_notrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_os_notrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-pt_file_ts_notrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_ts_notrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-pt_file_os_cstrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_os_cstrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-pt_file_ts_cstrp', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file_ts_cstrp', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-index_col', type=str, required=False, default=None,
                        help='index col to sort both baselines and latent df')
    parser.add_argument('-label_col', type=str, required=False, default='peptide',
                        help='column containing the labels (eg peptide)')
    parser.add_argument('-weight_col', type=str, required=False, default=None,
                        help='Column that contains the weight for a count (ex: norm_count); Leave empty to not use it')
    parser.add_argument('-rest_cols', type=str, required=False, default=None,
                        nargs='*', help='Other columns to be added; ex : -rest_cols peptide partition binder')
    parser.add_argument('-low_memory', type=str2bool, default=False,
                        help='whether to use "low memory merge mode. Might get wrong results...')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=1000,
                        help='How many points to do for the bounds limits')
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=None,
                        help='If provided, will skip the n_points iteration thing and run a single clustering at the given threshold.')
    parser.add_argument('-mp', '--min_purity', dest='min_purity', type=float, default=.8,
                        help='minimum purity for n_above')
    parser.add_argument('-ms', '--min_size', dest='min_size', type=int, default=6, help='minimum sizefor n_above')
    """
    TODO: Misc. 
    """
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-n_jobs', dest='n_jobs', default=-1, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    start = dt.now()
    print('Starting MST-cut pyscript')
    sns.set_style('darkgrid')
    args = vars(args_parser())
    unique_filename, kf, rid, connector = make_filename(args)

    # TODO: 
    # HERE NEED TO CHANGE OUTDIR
    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    # Here this is commented because we handle the uniquefilename creation already
    # in the overall bash script
    # outdir = os.path.join(outdir, unique_filename) + '/'
    # TODO : These things here need to change for a Webserver 
    mkdirs(outdir)
    # dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    # TODO HERE make sure columns etc are correct (ex A/B3 doesn't contain starting C/F etc.
    df = pd.read_csv(args['file'])
    # TODO : Hardcoded path or something server specific but the main directory would be in engine/src/tools/etc/models/
    # --> create directory structure to have each model saved in a separate folder and make loading easy
    # --> IF we need to do tcrbase etc maybe create a bash script that houses this code as embedded code ?
    model_paths = {'OSNOTRP': {'pt': ..., 'json': ...},
                   'OSCSTRP': {'pt': ..., 'json': ...},
                   'TSNOTRP': {'pt': ..., 'json': ...},
                   'TSCSTRP': {'pt': ..., 'json': ...}}
    assert args['model'] in model_paths.keys(), f"model provided is {args['model']} and is not in the keys of the dict!"
    model_paths = model_paths[args['model']]
    model = load_model_full(model_paths['pt'], model_paths['json'], map_location=args['device'], verbose=True)

    index_col = args['index_col']
    label_col = args['label_col']
    rest_cols = args['rest_cols']
    weight_col = args['weight_col']
    if weight_col is not None:
        if weight_col not in rest_cols and weight_col in df.columns:
            rest_cols.append(weight_col)

    latent_df = get_latent_df(model, df)

    if index_col is None or index_col == '' or index_col not in latent_df.columns:
        index_col = 'index_col'
        latent_df[index_col] = [f'seq_{i:04}' for i in range(len(latent_df))]
    seq_cols = tuple(args[f'{x.lower()}_col'] for x in ('A1', 'A2', 'A3', 'B1', 'B2', 'B3'))

    dist_matrix, dist_array, _, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(latent_df,
                                                                                                         label_col,
                                                                                                         seq_cols,
                                                                                                         index_col,
                                                                                                         rest_cols,
                                                                                                         args[
                                                                                                             'low_memory'])

    if args['threshold'] is not None:
        threshold = args['threshold']
    else:
        optimisation_results = agglo_all_thresholds(dist_array, dist_array, labels, encoded_labels, label_encoder, 5,
                                                    args['n_points'], args['min_purity'], args['min_size'], 'micro',
                                                    args['n_jobs'])
        optimisation_results['best'] = False
        optimisation_results.loc[
            optimisation_results.iloc[:int(0.8 * len(optimisation_results))]['silhouette'].idxmax(), 'best'] = True
        plot_sprm(optimisation_results, fn=f'{outdir}{unique_filename}optimisation_curves')
        threshold = optimisation_results.query('best')['threshold'].item()
        optimisation_results.to_csv(f'{outdir}{unique_filename}optimisation_results_df.csv')
    metrics, clusters_df, clusterer = agglo_single_threshold(dist_array, dist_array, labels, encoded_labels,
                                                             label_encoder, threshold,
                                                             'micro', args['min_purity'], args['min_size'],
                                                             return_df_and_c=True)
    clusters_df.to_csv(f'{outdir}{unique_filename}output_clusters_df.csv')
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'MST-cut finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')


if __name__ == '__main__':
    main()
