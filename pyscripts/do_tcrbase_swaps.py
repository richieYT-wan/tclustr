import glob

import pandas as pd
import os, sys

from tqdm.auto import tqdm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
from torch.utils.data import SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, mkdirs, get_random_id, get_datetime_string, get_class_initcode_keys
from src.torch_utils import load_model_full
from src.train_eval import predict_model
from src.datasets import FullTCRDataset
from src.models import TwoStageVAECLF, FullTCRVAE
from src.multimodal_models import BSSVAE, JMVAE
from src.multimodal_datasets import MultimodalPepTCRDataset
from src.multimodal_train_eval import predict_multimodal, embed_multimodal

from src.metrics import compute_cosine_distance
from src.sim_utils import make_dist_matrix
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from functools import partial


def sort_lines(data):
    # Split the data into lines
    lines = data.split('\n')

    # Sort the lines alphabetically based on the first column
    sorted_lines = sorted(lines, key=lambda x: x.split(':')[0])

    return sorted_lines


def do_tcrbase_and_histplots(preds, peptide, partition, f=None, ax=None,
                             unique_filename=None, outdir=None, bins=100):
    query = preds.query('partition==@partition and peptide==@peptide').assign(set='query')
    database = preds.query('partition!=@partition and peptide==@peptide and original_peptide==@peptide').assign(
        set='database')
    concat = pd.concat([query, database])
    dist_matrix = make_dist_matrix(concat, cols=('set', 'peptide', 'original_peptide', 'binder'))

    query = dist_matrix.query('set=="query"')
    database = dist_matrix.query('set=="database"')
    db_tcrs = database.index.tolist()
    # Scoring query against database & splitting by label
    pos = query[db_tcrs + ['binder']].query('binder==1')
    neg = query[db_tcrs + ['binder']].query('binder==0')
    tcrbase_output = pd.concat([pos, neg])
    pos = pos.drop(columns=['binder']).values
    neg = neg.drop(columns=['binder']).values

    # Getting the AUC for labelling and output DF ;
    pos_out = tcrbase_output.query('binder==1').drop(columns=['binder'])
    neg_out = tcrbase_output.query('binder!=1').drop(columns=['binder'])
    pos_out = pos_out.apply(lambda x: [np.min(x), x.index[int(np.argmin(x))]], axis=1, result_type='expand').rename(
        columns={0: 'min_dist', 1: 'most_similar'})
    neg_out = neg_out.apply(lambda x: [np.min(x), x.index[int(np.argmin(x))]], axis=1, result_type='expand').rename(
        columns={0: 'min_dist', 1: 'most_similar'})
    cat_out = pd.concat([pos_out.assign(label=1), neg_out.assign(label=0)])
    auc = roc_auc_score(cat_out['label'], 1 - cat_out['min_dist'])

    # Plot both the distribution of scores and the "best" score as done above
    #   Plotting distribution of all scores
    pos_flat = pos.flatten()
    neg_flat = neg.flatten()
    cat = np.concatenate([pos_flat, neg_flat])
    labels = np.concatenate([np.array(['pos'] * len(pos_flat) + ['neg'] * len(neg_flat))])
    df_plot_allvsall = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', 'label'])
    df_plot_allvsall['distance'] = df_plot_allvsall['distance'].astype(float)
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=df_plot_allvsall, x='distance', hue='label', ax=ax, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    # ax.set_xlim([0,1.1])
    ax.set_title(f'TCRBase: All vs All {peptide}: {auc:.4f}', fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f.savefig(f'{outdir}{peptide}_AvA_TCRBase_distances_histplot_{unique_filename}', dpi=150,
                  bbox_inches='tight')

    #   Plotting "Best" score
    pos_best = pos.min(axis=1).flatten()
    neg_best = neg.min(axis=1).flatten()
    cat_best = np.concatenate([pos_best, neg_best])
    labels_best = np.concatenate([np.array(['pos'] * len(pos_best) + ['neg'] * len(neg_best))])
    df_plot_best = pd.DataFrame(data=np.stack([cat_best, labels_best]).T, columns=['distance', 'label'])
    df_plot_best['distance'] = df_plot_best['distance'].astype(float)
    f2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
    bins = max(int(len(query.query('original_peptide==@peptide')) / 9), 25)

    sns.histplot(data=df_plot_best, x='distance', hue='label', ax=ax2, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    # ax.set_xlim([0,1.1])

    ax2.set_title(f'TCRBase: Best score {peptide}, AUC = {auc:.4f}', fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f2.savefig(f'{outdir}{peptide}_Best_TCRBase_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')

    return cat_out


def wrapper(preds, peptide, args, unique_filename, outdir):
    cat_out = do_tcrbase_and_histplots(preds, peptide, partition=args['fold'],
                                       unique_filename=unique_filename, outdir=outdir)

    cat_out.to_csv(f'{outdir}tcrbase_{peptide}_{unique_filename}.csv')

    auc = roc_auc_score(cat_out['label'], 1 - cat_out['min_dist'])
    auc01 = roc_auc_score(cat_out['label'], 1 - cat_out['min_dist'], max_fpr=0.1)
    text = f'\n{peptide}:\tauc={auc:.3f}\tauc01={auc01:.3f}'
    tqdm.write(text)

    return text


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to load a VAE model, extract similarity (or dist) and do TCRbase')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
                        help='Specify a device (cpu, cuda:0, cuda:1)')
    parser.add_argument('-db', '--db_file', dest='db_file', required=True, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input reference file')
    parser.add_argument('-qr', '--query_file', dest='query_file', type=str,
                        default=None, help='External test set (None by default). If None, will use "-kf"' \
                                           ' to split db_file into query (test) and db (train) files.' \
                                           ' If both are active, query_file will have priority')
    parser.add_argument('-kf', '--fold', dest='fold', type=int, default=1,
                        help='None by default. Will be used only if -qr is None. If -qr is a file, kf will be overriden.')
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
    parser.add_argument('-pep', '--pep_col', dest='pep_col', default='peptide', type=str, required=False,
                        help='Name of the column containing peptide sequences (inputs)')
    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=0,
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
    parser.add_argument('-pepenc', '--pep_encoding', dest='pep_encoding', type=str, default='categorical',
                        help='Which encoding to use for the peptide (onehot, BL50LO, BL62LO, BL62FREQ, categorical; Default = categorical)')
    parser.add_argument('-conv', dest='conv', type=str2bool, default=False,
                        help='Whether to use conv models and switch the dataset options')
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

    """
    Training hyperparameters & args
    """

    parser.add_argument('-dist_type', '--dist_type', dest='dist_type', default='cosine', type=str,
                        help='Which distance metric to use ')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    """
    TODO: Misc. 
    """
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=13,
                        help='Torch manual seed. Default = 13')
    parser.add_argument('-reset', dest='reset', type=str2bool, default=False,
                        help='Whether to reset the encoder\'s weight for a blank run')
    return parser.parse_args()


def main():
    start = dt.now()
    print('Starting script: ', start.strftime("%H:%M %d.%m.%y"))
    args = vars(args_parser())
    sns.set_style('darkgrid')

    # CUDA
    if torch.cuda.is_available() and args['cuda']:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if args['device'] is not None:
        device = args['device']
    print("Using : {}".format(device))

    # Handle the model first ; Assert then try to load it
    assert not all([args[k] is None for k in ['model_folder', 'pt_file', 'json_file']]), \
        'Please provide either the path to the folder containing the .pt and .json or paths to each file (.pt/.json) separately!'
    connector = '' if args["out"] == '' else '_'
    if args['model_folder'] is not None:
        try:
            checkpoint_file = next(
                filter(lambda x: 'best' in x.lower() and 'checkpoint' in x and 'interval' not in x and 'last' not in x,
                       glob.glob(f'{args["model_folder"]}/*.pt')))
            checkpoint_file = checkpoint_file if args['pt_file'] is None else args['pt_file']
            model_json = next(filter(lambda x: 'checkpoint' in x,
                                     glob.glob(f'{args["model_folder"]}/*.json'))) if args['json_file'] is None else args['json_file']
            vae, js = load_model_full(checkpoint_file, model_json,
                                      return_json=True, map_location=device)
        except:
            print(args['model_folder'], os.listdir(args['model_folder']))
            raise ValueError(f'\n\nCouldn\'t load your files! at {args["model_folder"]}\n\n')
    else:
        vae, js = load_model_full(args['pt_file'], args['json_file'], return_json=True, map_location=device)

    # here, extracts the VAE if it's part of the Bimodal
    if isinstance(vae, TwoStageVAECLF) and hasattr(vae, 'vae'):
        vae = vae.vae
        js = js["vae_kwargs"]

    if args['reset']:
        vae.reset_parameters(seed=args['seed'])
    # print(js, vae)
    # Checking whether we have a query (test) file or using a kf to split the train dataframe
    # and read the data
    assert not all(args[k] is None for k in ['query_file', 'fold']), \
        'No query file was provided and kf is set to None. Need to use either an external test file or provide a kf (partition)!'
    if args['query_file'] is None:
        df = pd.read_csv(args['db_file'])
        db_df = df.query('partition!=@args["fold"]')
        qr_df = df.query('partition==@args["fold"]')
    else:
        db_df = pd.read_csv(args['db_file'])
        qr_df = pd.read_csv(args['qr_file'])
        args['fold'] = -1

    kf = args['fold']

    # Creating random_id and output_directories
    rid = args['random_id'] if (args['random_id'] is not None and args['random_id'] != '') else \
        get_random_id() if args['random_id'] == '' else \
            args['random_id']

    unique_filename = f'{args["out"]}{connector}KFold_{kf}_{get_datetime_string()}_{rid}'
    outdir = '../output/'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'
    outdir = os.path.join(outdir, unique_filename) + '/'
    mkdirs(outdir)

    # Loading args & dataset instances
    for k in args:
        if 'max_len' in k or 'positional' in k:
            args[k] = js[k] if k in js else 0

    if isinstance(vae, JMVAE) or isinstance(vae, BSSVAE):
        dataset_keys = get_class_initcode_keys(MultimodalPepTCRDataset, args)
        dataset_params = {k: args[k] for k in dataset_keys}
        # TODO: for convenience do pair only and return pair only here
        dataset_params['pair_only'] = True
        dataset_params['return_pair'] = True
        db_dataset = MultimodalPepTCRDataset(db_df, **dataset_params)
        qr_dataset = MultimodalPepTCRDataset(qr_df, **dataset_params)
        db_loader = db_dataset.get_dataloader(batch_size=2048, sampler=SequentialSampler)
        qr_loader = qr_dataset.get_dataloader(batch_size=2048, sampler=SequentialSampler)
        db_preds = embed_multimodal(vae, db_dataset, db_loader)
        qr_preds = embed_multimodal(vae, qr_dataset, qr_loader)

    else:
        dataset_keys = get_class_initcode_keys(FullTCRDataset, args)
        dataset_params = {k: args[k] for k in dataset_keys}
        db_dataset = FullTCRDataset(db_df, **dataset_params)
        qr_dataset = FullTCRDataset(qr_df, **dataset_params)
        db_loader = db_dataset.get_dataloader(batch_size=2048, sampler=SequentialSampler)
        qr_loader = qr_dataset.get_dataloader(batch_size=2048, sampler=SequentialSampler)
        # Get the "predictions" (latent vectors)
        db_preds = predict_model(vae, db_dataset, db_loader)
        qr_preds = predict_model(vae, qr_dataset, qr_loader)

    # Here, concatenate and construct the distance matrix a single time ; Keep the columns for filtering later
    concat = pd.concat([qr_preds.assign(set="query"), db_preds.assign(set="database")])
    concat['tcr'] = concat.apply(lambda x: ''.join([x[c] for c in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']]), axis=1)

    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    wrapper_ = partial(wrapper, preds=concat, args=args, unique_filename=unique_filename, outdir=outdir)
    # Then, on a per-peptide basis, do the TCRbase method
    text = Parallel(n_jobs=6)(
        delayed(wrapper_)(peptide=peptide) for peptide in tqdm(concat.peptide.unique(), desc='peptide'))

    text = ''.join(text)
    text = sort_lines(text)
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        for line in text:
            file.write(f'{line}\n')


if __name__ == '__main__':
    main()
