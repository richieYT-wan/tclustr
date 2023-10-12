import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import argparse
import copy
import torch
from torch import optim
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists
from src.torch_utils import load_checkpoint
from src.models import CDR3bVAE
from src.train_eval import predict_model, train_eval_loops
from src.datasets import CDR3BetaDataset
from src.metrics import VAELoss, get_metrics
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
import colorcet as cc
from src.utils import get_palette
from matplotlib import pyplot as plt
import seaborn as sns


def args_parser():
    parser = argparse.ArgumentParser(
        description='Do distance extraction and clustering for all folds contained in a maindirectory')
    """
    Data processing args
    """
    parser.add_argument('-d', '--dir', dest='dir', required=False,
                        default='/home/projects/vaccine/people/yatwan/tclustr/output/30K_epochs_OnlyPositivesFullCDR3b_LowerDim_64_WD_1e-4_EDHpH/',
                        type=str, help='directory containing all 5 folds sub-directories')
    parser.add_argument('-f', '--file', dest='file', required=False,
                        default='/home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv',
                        type=str, help='train file')
    parser.add_argument('-o', '--outdir', dest='outdir', required=False,
                        type=str, default='/home/projects/vaccine/people/yatwan/tclustr/output/231012_redo_clusters/')

    return parser.parse_args()


def reassign_label_duplicates(cdr3b, df):
    max_label = df.query('TRB_CDR3==@cdr3b').groupby('peptide').agg(count=('binder', 'count')).idxmax().item()
    return max_label


def get_cluster_stats(input_df, cluster='KMeans_Cluster', label='GroundTruth', feature='TSNE_1', kf=True):
    if kf:
        summaries = []
        dfs = []
        for fold in input_df.partition.unique():
            df = input_df.query('partition==@fold')
            l = len(df)
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
            summary['retention'] = (100 * summary['cluster_size']) / l
            df['partition'] = fold
            summaries.append(summary)
            dfs.append(df)
        return pd.concat(summaries), pd.concat(dfs)
    else:
        df = input_df
        l = len(df)
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
        summary['retention'] = 100 * summary['cluster_size'] / l
        return summary, df


def get_good_clusters(summary_df, percent=50, size=3, dict=False):
    tmp = summary_df.query('purity_percent>@percent and cluster_size>=@size')
    results = {'n_clusters': len(tmp), 'n_total_tcrs': tmp.cluster_size.sum(),
               'mean_cluster_size': tmp.cluster_size.mean(), 'mean_purity': tmp.purity_percent.mean()}
    if dict:
        return results
    else:
        return list(results.values())


def get_good_summary(summary, purity_threshold=60, size_threshold=3):
    gb = summary.groupby(['metric', 'linkage'])
    return pd.DataFrame(data=np.stack(gb.apply(get_good_clusters, purity_threshold, size_threshold).values).round(3),
                        index=gb.groups.keys(),
                        columns=['n_clusters', 'n_total_tcrs', 'mean_cluster_size', 'mean_purity'])


def main():
    args = vars(args_parser())
    maindir = args['dir']
    outdir = args['outdir']
    os.makedirs(outdir, exist_ok=True)

    fold_dirs = sorted([f'{maindir}{subdir}/' for subdir in os.listdir(maindir)])
    df = pd.read_csv(args['file'])
    df['labels'] = df['TRB_CDR3'].apply(reassign_label_duplicates, df=df)
    df.drop_duplicates(subset=['TRB_CDR3'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['seq_id'] = [f'seq_{i:04}' for i in range(len(df))]
    #  INIT PARAMS
    max_len = 25
    encoding = 'BL50LO'
    pad_scale = -20
    use_v = False
    use_j = False
    v_dim = 0
    j_dim = 0
    activation = nn.SELU()
    hidden_dim = 128
    latent_dim = 64
    zcols = [f'z_{i}' for i in range(latent_dim)]
    max_len_pep = 0
    aa_dim = 20
    for i, fd in enumerate(fold_dirs):
        basename = os.path.basename(fd)
        checkpoint = f"{fd}{next(filter(lambda x: 'checkpoint' in x and x.endswith('.pt'), os.listdir(fd)))}"
        train = df.query('partition!=@i')
        valid = df.query('partition==@i')
        model = CDR3bVAE(max_len, encoding, pad_scale, aa_dim, use_v, use_j, v_dim, j_dim, activation, hidden_dim,
                         latent_dim, max_len_pep)
        model = load_checkpoint(model, checkpoint)
        train_dataset = CDR3BetaDataset(train, max_len, encoding, pad_scale, 'TRB_CDR3', use_v, use_j, None, None,
                                        v_dim, j_dim, None, None, False, max_len_pep)
        train_loader = train_dataset.get_dataloader(1024, SequentialSampler)
        valid_dataset = CDR3BetaDataset(valid, max_len, encoding, pad_scale, 'TRB_CDR3', use_v, use_j, None, None,
                                        v_dim, j_dim, None, None, False, max_len_pep)
        valid_loader = valid_dataset.get_dataloader(1024, SequentialSampler)
        train_preds = predict_model(model, train_dataset, train_loader).assign(set='train')
        valid_preds = predict_model(model, valid_dataset, valid_loader).assign(set='valid')
        cat = pd.concat([train_preds, valid_preds])

        z_values = cat[zcols].values
        seq_idx_cols = cat.TRB_CDR3.values
        labels = np.expand_dims(cat.labels.values, 1)
        ids = np.expand_dims(cat.seq_id.values, 1)
        sets = np.expand_dims(cat.set.values, 1)
        distance_matrices = {}
        best_metric = 0
        best_df, best_summ, best_gs = None, None, None
        gs_dfs = []
        for distance in ['manhattan', 'euclidean', 'cosine', 'correlation']:
            out_dist = pairwise_distances(X=z_values, metric=distance)
            dist_matrix = pd.DataFrame(np.concatenate([out_dist, labels, ids, sets], axis=1),
                                       index=seq_idx_cols, columns=list(seq_idx_cols) + ['labels', 'ids', 'set'])
            dist_matrix.to_csv(f'{fd}{basename}_distance_{distance}.csv')
            distance_matrices[distance] = dist_matrix
            for linkage in ['average', 'complete']:
                for n_clusters in [635, 878, 1000, 1500, 2000, 2250]:
                    ag = AgglomerativeClustering(metric='precomputed', linkage=linkage, n_clusters=n_clusters,
                                                 distance_threshold=None, compute_distances=True)
                    pred_clusters = ag.fit_predict(dist_matrix.drop(columns=['labels', 'ids', 'set']))

                    ag_df = pd.DataFrame(np.stack([dist_matrix.index, pred_clusters, labels, ids, sets], axis=1),
                                         columns=['TRB_CDR3', 'pred_cluster', 'labels', 'ids', 'set'])
                    ag_df = pd.merge(cat.rename(columns={'seq_id': 'ids'}), ag_df,
                                     left_on=['TRB_CDR3', 'labels', 'ids', 'set'],
                                     right_on=['TRB_CDR3', 'labels', 'ids', 'set'])
                    summary = \
                    get_cluster_stats(ag_df, cluster='pred_cluster', label='labels', feature='TRB_CDR3', kf=False)[0]
                    summary['metric'] = distance
                    summary['linkage'] = linkage
                    summary['nc'] = n_clusters
                    summary['basename'] = basename
                    gs_df = get_good_clusters(summary, percent=75, size=4).dropna() \
                        .sort_values(['n_total_tcrs', 'mean_purity'], ascending=False).reset_index()
                    gs_df['nc'] = n_clusters
                    gs_df['retention'] = gs_df['n_total_tcrs'] / len(cat)
                    gs_df['agg_metric'] = .5 * (gs_df['retention'] + gs_df['mean_purity'])
                    if gs_df['agg_metric'].item() > best_metric:
                        best_df = ag_df
                        best_summ = summary
                        best_gs = gs_df
                    ag_df.to_csv(f'{fd}{basename}_ag_df.csv')
                    summary.to_csv(f'{fd}{basename}_summary_df.csv')
                    gs_df.to_csv(f'{fd}{basename}_good_df.csv')
                    gs_dfs.append(gs_df)
        best_agdf = pd.merge(best_df, best_summ.rename(columns={'labels': 'cluster_label'})[
            ['pred_cluster', 'cluster_label', 'purity_percent', 'cluster_size']],
                                    left_on=['pred_cluster'], right_on=['pred_cluster'])

        best_train = best_agdf.query('set=="train"')
        best_valid = best_agdf.query('set=="valid"')
        z_train = best_train[zcols].values
        z_valid = best_valid[zcols].values
        labels_train = train['peptide'].values
        labels_valid = valid['peptide'].values

        tsne = TSNE(n_components=2, n_iter=1500, metric='cosine', perplexity=30)
        latent_tsne = tsne.fit_transform(np.concatenate([z_train, z_valid], axis=0))
        tsne_train, tsne_valid = latent_tsne[:len(z_train)], latent_tsne[len(z_train):]
        best_train[['TSNE_1', 'TSNE_2']] = tsne_train
        best_train['pred_label'] = best_train.apply(lambda x: x['cluster_label'] if x['purity_percent'] >= 70 else 'trash',
                                          axis=1)
        f, a = plt.subplots(1, 2, figsize=(25, 12))
        a = a.ravel()

        pep_order = sorted(df.peptide.unique()) + ['trash']

        sns.scatterplot(data=best_train,  # .query('GroundTruth!="immrep_negs"'),
                        x='TSNE_1', y='TSNE_2', s=13, hue='pred_label', hue_order=pep_order, ax=a[0])
        sns.scatterplot(data=best_train,  # .query('GroundTruth!="immrep_negs"'),
                        x='TSNE_1', y='TSNE_2', s=13, hue='peptide', hue_order=pep_order, ax=a[1])

        a[0].set_title('Train : Predicted clusters', fontsize=24, fontweight='semibold')
        a[1].set_title('Train : Ground Truth', fontsize=24, fontweight='semibold')

        f.suptitle(f'TRAIN: fold={i}t-SNE Visualization with K-Means Clusters on Latent (d=64) ; Only TRB_CDR3\n'
                   f'using {best_gs.index}',
                   fontsize=26, fontweight='semibold')
        f.tight_layout()
        f.savefig(f'{fd}tsne_plot_train.png', bbox_inches='tight', dpi=300)
        best_valid[['TSNE_1', 'TSNE_2']] = tsne_valid
        best_valid['pred_label'] = best_valid.apply(lambda x: x['cluster_label'] if x['purity_percent'] >= 70 else 'trash',
                                          axis=1)
        f, a = plt.subplots(1, 2, figsize=(25, 12))
        a = a.ravel()

        pep_order = sorted(df.peptide.unique()) + ['trash']

        sns.scatterplot(data=best_valid,  # .query('GroundTruth!="immrep_negs"'),
                        x='TSNE_1', y='TSNE_2', s=13, hue='pred_label', hue_order=pep_order, ax=a[0])
        sns.scatterplot(data=best_valid,  # .query('GroundTruth!="immrep_negs"'),
                        x='TSNE_1', y='TSNE_2', s=13, hue='peptide', hue_order=pep_order, ax=a[1])

        a[0].set_title('Train : Predicted clusters', fontsize=24, fontweight='semibold')
        a[1].set_title('Train : Ground Truth', fontsize=24, fontweight='semibold')

        f.suptitle(f'VALID: fold={i}\nt-SNE Visualization with K-Means Clusters on Latent (d=64) ; Only TRB_CDR3\n'
                   f'using {best_gs.index}',
                   fontsize=26, fontweight='semibold')
        f.tight_layout()
        f.savefig(f'{fd}tsne_plot_valid.png', bbox_inches='tight', dpi=300)



if __name__=='__main__':
    main()

