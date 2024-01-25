import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from src.metrics import compute_cosine_distance


def make_dist_matrix(df, seq_cols=('A1', 'A2', 'A3', 'B1', 'B2', 'B3')):
    df['seq'] = df.apply(lambda x: ''.join([x[c] for c in seq_cols]), axis=1)
    seqs = df.seq.values
    # Getting dist matrix
    zcols = [z for z in df.columns if z.startswith("z_")]
    zs = torch.from_numpy(df[zcols].values)
    dist_matrix = pd.DataFrame(compute_cosine_distance(zs),
                               columns=seqs, index=seqs)
    dist_matrix = pd.merge(dist_matrix, df.set_index('seq')[['set', 'peptide', 'original_peptide']],
                           left_index=True, right_index=True).rename(columns={'peptide': 'label'})
    return dist_matrix


def get_tcrbase_method(tcr, ref):
    # here take the top 1 (shorted distance = highest sim)
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


def do_histplot(dist_matrix, peptide, label_col='label', unique_filename=None, outdir=None):
    # splitting the matrix by label
    dist_matrix = dist_matrix[dist_matrix.index.tolist() + ['set', label_col, 'original_peptide']]
    same = dist_matrix.query(f'{label_col}==@peptide')
    same_tcrs = same.index.tolist()
    diff = dist_matrix.query(f'{label_col}!=@peptide')
    diff_tcrs = diff.index.tolist()
    same_matrix = same[same_tcrs].values
    diff_matrix = same[diff_tcrs].values
    # Getting the flattened distributions (upper triangle and making df for plot)
    trimask = np.triu(np.ones(same_matrix.shape), k=1)
    masked_same = np.multiply(same_matrix, trimask)
    flattened_same = masked_same[masked_same != 0].flatten()
    flattened_diff = diff_matrix.flatten()
    cat = np.concatenate([flattened_same, flattened_diff])
    labels = np.concatenate([np.array(['same'] * len(flattened_same) + ['diff'] * len(flattened_diff))])
    ntr = pd.DataFrame(data=np.stack([cat, labels]).T, columns=['distance', label_col])
    ntr['distance'] = ntr['distance'].astype(float)
    # plotting
    pal = sns.color_palette('gnuplot2', 4)
    sns.set_palette([pal[-1], pal[0]])
    sns.set_style('darkgrid')
    f, a = plt.subplots(1, 1, figsize=(9, 5))
    sns.histplot(data=ntr, x='distance', hue=label_col, ax=a, kde=False,
                 stat='percent', common_norm=False, bins=300, alpha=0.75)
    a.set_title(peptide, fontsize=14, fontweight='semibold')
    if unique_filename is not None:
        outdir = './' if outdir is None else outdir
        f.savefig(f'{outdir}{peptide}_distances_histplot_{unique_filename}', dpi=150, bbox_inches='tight')


def wrapper(dist_matrix, peptide, unique_filename, outdir):
    query = dist_matrix.query('set=="query"').copy()
    database = dist_matrix.query('set=="database" and label==@peptide').copy()
    output = do_tcrbase(query, database, label=peptide)
    output.to_csv(f'{outdir}tcrbase_{peptide}_{unique_filename}.csv')
    auc = roc_auc_score(output['y_true'], output['score'])
    auc01 = roc_auc_score(output['y_true'], output['score'], max_fpr=0.1)
    text = f'\n{peptide}:\tauc={auc:.3f}\tauc01={auc01:.3f}'
    do_histplot(dist_matrix, peptide, unique_filename, outdir)
    tqdm.write(text)
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'{text}')
