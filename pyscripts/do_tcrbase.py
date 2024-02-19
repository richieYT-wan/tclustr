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
from src.datasets import FullTCRDataset, BimodalTCRpMHCDataset
from src.models import BimodalVAEClassifier, FullTCRVAE
from src.metrics import compute_cosine_distance
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


def get_tcrbase_method(tcr, ref):
    # here take the top 1 (shortest distance = highest sim)
    best = ref[tcr].sort_values().head(1)
    best_name = best.index[0]
    best_dist = best.item()
    label = ref.loc[best_name]['label']
    return label, best_name, best_dist


def do_tcrbase(query_distmatrix, db_distmatrix, label='GILGFVFTL'):
    """
    dist_matrix is the filtered query matrix
    ref is the filtered database matrix
    Args:
        query_distmatrix:
        db_distmatrix:
        label:

    Returns:

    """
    output = query_distmatrix.drop(
        columns=[x for x in query_distmatrix.columns if x != 'label']).copy().reset_index().rename(
        columns={'index': 'tcr'})

    output[['similar_label', 'best_match', 'best_dist']] = output.apply(
        lambda x: get_tcrbase_method(x['tcr'], ref=db_distmatrix),
        axis=1, result_type='expand')
    output['y_true'] = (output['label'] == label).astype(int)
    output['score'] = 1 - output['best_dist']
    return output.sort_values(['y_true', 'score'], ascending=False)


def do_histplot(dist_matrix, peptide, unique_filename, outdir, auc=None):
    # splitting the matrix by label
    valid = dist_matrix.query('set=="query"')
    valid = valid[valid.index.tolist() + ['set', 'label', 'original_peptide']]
    same = valid.query('label==@peptide')
    same_tcrs = same.index.tolist()
    diff = valid.query('label!=@peptide')
    diff_tcrs = diff.index.tolist()
    same_matrix = same[same_tcrs].values
    diff_matrix = same[diff_tcrs].values
    # Getting the flattened distributions (upper triangle and making df for plot), for AvA
    trimask = np.triu(np.ones(same_matrix.shape), k=1)
    masked_same = np.multiply(same_matrix, trimask)
    flattened_same = masked_same[masked_same != 0].flatten()
    flattened_diff = diff_matrix.flatten()
    cat = np.concatenate([flattened_same, flattened_diff])
    labels = np.concatenate([np.array(['same'] * len(flattened_same) + ['diff'] * len(flattened_diff))])
    ntr = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', 'label'])
    ntr['distance'] = ntr['distance'].astype(float)
    # plotting
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')
    f, a = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=ntr, x='distance', hue='label', ax=a, kde=False, stat='percent', common_norm=False, bins=100,
                 alpha=0.75)
    a.set_title(f'All vs All {peptide}: AUC={auc:.4f}', fontsize=14, fontweight='semibold')
    f.savefig(f'{outdir}{peptide}_AvA_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')

    # Here, plotting the best score only
    f2, ax2 = plt.subplots(1,1, figsize=(9,5))
    same_best = same_matrix.min(axis=1).flatten()
    diff_best = diff_matrix.min(axis=1).flatten()
    cat_best = np.concatenate([same_best, diff_best])
    labels_best = np.concatenate([np.array(['same'] * len(same_best) + ['diff'] * len(diff_best))])
    df_plot_best = pd.DataFrame(data=np.stack([cat_best, labels_best]).T, columns=['distance', 'label'])
    df_plot_best['distance'] = df_plot_best['distance'].astype(float)

    bins = max(int(len(dist_matrix.query('original_peptide==@peptide'))/9), 25)
    sns.histplot(data=df_plot_best, x='distance', hue='label', ax=ax2, kde=False,
                 stat='percent', common_norm=False, bins=bins, alpha=0.75)
    ax2.set_title(f'Best {peptide}, AUC = {auc:.4f}', fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f2.savefig(f'{outdir}{peptide}_Best_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')


def wrapper(dist_matrix, peptide, unique_filename, outdir):
    query = dist_matrix.query('set=="query"').copy()
    database = dist_matrix.query('set=="database" and label==@peptide').copy()
    output = do_tcrbase(query, database, label=peptide)
    output.to_csv(f'{outdir}tcrbase_{peptide}_{unique_filename}.csv')
    auc = roc_auc_score(output['y_true'], output['score'])
    auc01 = roc_auc_score(output['y_true'], output['score'], max_fpr=0.1)
    text = f'\n{peptide}:\tauc={auc:.3f}\tauc01={auc01:.3f}'
    do_histplot(dist_matrix, peptide, unique_filename, outdir, auc=auc)
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

    parser.add_argument('-db', '--db_file', dest='db_file', required=True, type=str,
                        default='../data/filtered/230927_nettcr_positives_only.csv',
                        help='filename of the input reference file')
    parser.add_argument('-qr', '--query_file', dest='query_file', type=str,
                        default=None, help='External test set (None by default). If None, will use "-kf"' \
                                           ' to split db_file into query (test) and db (train) files.' \
                                           ' If both are active, query_file will have priority')
    parser.add_argument('-kf', '--fold', dest='fold', type=int, default=0,
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
    return parser.parse_args()


def main():
    start = dt.now()
    print('Starting script: ', start.strftime("%H:%M %d.%m.%y"))
    args = vars(args_parser())
    sns.set_style('darkgrid')
    # Handle the model first ; Assert then try to load it
    assert not all([args[k] is None for k in ['model_folder', 'pt_file', 'json_file']]), \
        'Please provide either the path to the folder containing the .pt and .json or paths to each file (.pt/.json) separately!'
    connector = '' if args["out"] == '' else '_'
    if args['model_folder'] is not None:
        try:
            checkpoint_file = next(
                filter(lambda x: x.startswith('checkpoint') and x.endswith('.pt'), os.listdir(args['model_folder'])))
            json_file = next(
                filter(lambda x: x.startswith('checkpoint') and x.endswith('.json'), os.listdir(args['model_folder'])))
            vae, js = load_model_full(args['model_folder'] + checkpoint_file, args['model_folder'] + json_file,
                                      return_json=True)
        except:
            print(args['model_folder'], os.listdir(args['model_folder']))
            raise ValueError(f'\n\nCouldn\'t load your files! at {args["model_folder"]}\n\n')
    else:
        vae, js = load_model_full(args['pt_file'], args['json_file'], return_json=True)

    # here, extracts the VAE if it's part of the Bimodal
    if isinstance(vae, BimodalVAEClassifier) and hasattr(vae, 'vae'):
        vae = vae.vae
        js = js["vae_kwargs"]

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

    # CUDA
    if torch.cuda.is_available() and args['cuda']:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print("Using : {}".format(device))

    # Loading args & dataset instances
    for k in args:
        if 'max_len' in k or 'positional' in k:
            args[k] = js[k] if k in js else 0

    dataset_keys = get_class_initcode_keys(FullTCRDataset, args)
    dataset_params = {k: args[k] for k in dataset_keys}
    db_dataset = FullTCRDataset(db_df, **dataset_params)
    qr_dataset = FullTCRDataset(qr_df, **dataset_params)
    db_loader = db_dataset.get_dataloader(batch_size=1024, sampler=SequentialSampler)
    qr_loader = qr_dataset.get_dataloader(batch_size=1024, sampler=SequentialSampler)

    # Get the "predictions" (latent vectors)
    db_preds = predict_model(vae, db_dataset, db_loader)
    qr_preds = predict_model(vae, qr_dataset, qr_loader)
    # Here, concatenate and construct the distance matrix a single time ; Keep the columns for filtering later
    concat = pd.concat([qr_preds.assign(set="query"), db_preds.assign(set="database")])
    concat['tcr'] = concat.apply(lambda x: ''.join([x[c] for c in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']]), axis=1)
    tcrs = concat.tcr.values
    # Getting dist matrix
    zcols = [z for z in concat.columns if z.startswith("z_")]
    zs = torch.from_numpy(concat[zcols].values)
    dist_matrix = pd.DataFrame(compute_cosine_distance(zs),
                               columns=tcrs, index=tcrs)
    dist_matrix = pd.merge(dist_matrix, concat.set_index('tcr')[['set', 'peptide', 'original_peptide']],
                           left_index=True, right_index=True).rename(columns={'peptide': 'label'})

    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    wrapper_ = partial(wrapper, dist_matrix=dist_matrix, unique_filename=unique_filename, outdir=outdir)
    # Then, on a per-peptide basis, do the TCRbase method
    text = Parallel(n_jobs=8)(
        delayed(wrapper_)(peptide=peptide) for peptide in tqdm(dist_matrix.label.unique(), desc='peptide'))
    text = ''.join(text)
    text = sort_lines(text)
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        for line in text:
            file.write(f'{line}\n')


if __name__ == '__main__':
    main()
