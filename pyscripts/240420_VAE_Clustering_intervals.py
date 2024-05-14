import pandas as pd
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string
import argparse
from src.cluster_utils import resort_baseline, do_baseline, run_interval_plot_pipeline


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
    """
    Training hyperparameters & args
    """

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-np', '--n_points', dest='n_points', type=int, default=500,
                        help='How many points to do for the bounds limits')

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
    return parser.parse_args()


def main():
    print('Starting script')
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    assert not all([args[k] is None for k in ['model_folder', 'pt_file', 'json_file']]), \
        'Please provide either the path to the folder containing the .pt and .json or paths to each file (.pt/.json) separately!'
    connector = '' if args["out"] == '' else '_'
    kf = '-1' if args["fold"] is None else args['fold']
    rid = args['random_id'] if (args['random_id'] is not None and args['random_id'] != '') else get_random_id() if args[
                                                                                                                       'random_id'] == '' else \
        args['random_id']

    dfname = args['file'].split('/')[-1].split('.')[0]

    unique_filename = f'{args["out"]}{connector}{dfname}_{get_datetime_string()}_{rid}'

    outdir = '../output/'
    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'

    outdir = os.path.join(outdir, unique_filename) + '/'

    df = pd.read_csv(args['file'])
    if args['index_col'] is not None:
        assert args['index_col'] in df.columns, f'Provided index_col {args["index_col"]} not in columns!'
        index_col = args['index_col']
    else:
        assert 'raw_index' in df.columns or 'original_index' in df.columns, 'Index col not in df! (neither raw_index or original_index)'
        index_col = 'raw_index' if 'raw_index' in df.columns else 'original_index'

    tbcralign = pd.read_csv(args['tbcralign_file'], index_col=0)
    tcrdist = pd.read_csv(args['tcrdist_file'], index_col=0)

    mkdirs(outdir)
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    if args['fold'] is not None:
        kf = args['fold']
        train_baselines = []
        valid_baselines = []
        # Here need to re-sort baseline based on the input, in case we are trying to cluster reduced versions of the datasets
        train_df = df.query('partition!=@kf')
        valid_df = df.query('partition==@kf')
        tbcralign_train, _ = resort_baseline(tbcralign.query('partition!=@kf'), train_df, index_col)
        tcrdist_train, _ = resort_baseline(tcrdist.query('partition!=@kf'), train_df, index_col)
        tbcralign_valid, _ = resort_baseline(tbcralign.query('partition==@kf'), valid_df, index_col)
        tcrdist_valid, _ = resort_baseline(tcrdist.query('partition==@kf'), valid_df, index_col)
        train_baselines.append(do_baseline(tbcralign_train, identifier='TBCRalign', label_col=args['label_col'],
                                           n_points=args['n_points']))
        train_baselines.append(
            do_baseline(tcrdist_train, identifier='tcrdist3', label_col=args['label_col'], n_points=args['n_points']))
        valid_baselines.append(do_baseline(tbcralign_valid, identifier='TBCRalign', label_col=args['label_col'],
                                           n_points=args['n_points']))
        valid_baselines.append(
            do_baseline(tcrdist_valid, identifier='tcrdist3', label_col=args['label_col'], n_points=args['n_points']))
        train_baselines = pd.concat(train_baselines)
        valid_baselines = pd.concat(valid_baselines)
        train_results = run_interval_plot_pipeline(args['model_folder'], train_df, index_col, args['label_col'],
                                                   tbcralign_train, args['out'], args['n_points'], train_baselines,
                                                   f"Train {args['out']}", outdir + unique_filename + 'train_pur_ret_curve')
        valid_results = run_interval_plot_pipeline(args['model_folder'], valid_df, index_col, args['label_col'],
                                                   tbcralign_valid, args['out'], args['n_points'], valid_baselines,
                                                   f"Valid {args['out']}", outdir + unique_filename + 'valid_pur_ret_curve')

        train_results.to_csv(f'{outdir}{unique_filename}_train_results.csv')
        valid_results.to_csv(f'{outdir}{unique_filename}_valid_results.csv')
    # TODO : when not None
    else:
        baselines = [do_baseline(tbcralign,
                                 identifier='TBCRalign',
                                 label_col=args['label_col'], n_points=args['n_points']),
                     do_baseline(tcrdist, identifier='tcrdist3', label_col=args['label_col'],
                                 n_points=args['n_points'])]
        results = run_interval_plot_pipeline(args['model_folder'], df, index_col, args['label_col'],
                                             tbcralign, args['out'], args['n_points'], baselines, args['out'],
                                             outdir + unique_filename + 'pur_ret_curve')
        results.to_csv(f'{outdir}{unique_filename}_results.csv')

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
