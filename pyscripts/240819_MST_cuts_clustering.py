import argparse
import pandas as pd
from src.cluster_utils import *
from src.networkx_utils import *
from src.torch_utils import load_model_full
from src.utils import str2bool, make_filename


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
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
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory')
    parser.add_argument('-pt_file', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    parser.add_argument('-index_col', type=str, required=False, default=None,
                        help='index col to sort both baselines and latent df')
    parser.add_argument('-label_col', type=str, required=False, default='peptide',
                        help='column containing the labels (eg peptide)')
    parser.add_argument('-rb', type=str2bool, required=False, dest='reload_baselines', default=None,
                        help='True/False to reload baselines')
    parser.add_argument('-bf', type=str, required=False, dest='baselines_folder', default=None,
                        help='path containing the baselines (eg peptide)')
    parser.add_argument('-dn', type=str, required=False, dest='dataset_name', default=None,
                        help="name of dataset of baseline to use ['OldDataTop15', 'ExpDataTop78', 'OldDataTop20', 'OldDataNoPrune','ExpData17peps']")
    """
    Training hyperparameters & args
    """

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=500,
                        help='How many points to do for the bounds limits')
    parser.add_argument('-link', '--linkage', dest='linkage', type=str, default='complete',
                        help='Which linkage to use for AgglomerativeClustering')
    """
    TODO: Misc. 
    """
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    parser.add_argument('-reset', dest='reset', type=str2bool, default=False,
                        help='Whether to reset the encoder\'s weight for a blank run')
    parser.add_argument('-random_latent', dest='random_latent', type=str2bool, default=False,
                        help='Whether to set RANDOM latent vectors')
    parser.add_argument('-newmodel', dest='newmodel', type=str2bool, default=False,
                        help='re instanciate a new model from scratch')
    parser.add_argument('-tcr_enc', dest='tcr_enc', type=str, default=None,
                        help='Whether to do "alternative" TCR encoding')
    parser.add_argument('-n_jobs', dest='n_jobs', default=1, type=int,
                        help='Multiprocessing')
    return parser.parse_args()


def main():
    args = vars(args_parser())
    unique_filename, kf, rid, connector = make_filename(args)
    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)

    df = pd.read_csv(args['file'])
    idxs = df[args['index_col']].unique()
    model = load_model_full(args['pt_file'], args['json_file'], map_location=args['device'], verbose=False)
    latent_df = get_latent_df(model, df)
    dist_matrix, dist_array, features, labels, encoded_labels, label_encoder = get_distances_labels_from_latent(latent_df, index_col=args['index_col'])

    if args['do_baselines']:
        dist_matrix_tbcralign = pd.read_csv(args['tbcr_file'])
        dist_matrix_tcrdist3 = pd.read_csv(args['tcrdist_file'])
        dist_matrix_tbcralign, values_tbcralign = resort_baseline(dist_matrix_tbcralign, dist_matrix, args["index_col"])
        dist_matrix_tcrdist3, values_tcrdist3 = resort_baseline(dist_matrix_tcrdist3, dist_matrix, args["index_col"])




if __name__ == '__main__':
    main()