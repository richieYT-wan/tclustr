from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
from sklearn.cluster import *
import numpy as np
import pandas as pd
from datetime import datetime as dt
import os


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


def bruteforce(eps, metric, ODIR, z_train_30k, z_valid_30k, z_gil_30k, train_preds_30k, valid_preds_30k, gil_preds_30k):
    db = DBSCAN(eps=eps, metric=metric, n_jobs=9)
    clusters = db.fit_predict(np.concatenate([z_train_30k, z_valid_30k, z_gil_30k], axis=0))
    train_clusters = clusters[:len(z_train_30k)]
    valid_clusters = clusters[len(z_train_30k): len(z_train_30k) + len(z_valid_30k)]
    gil_clusters = clusters[len(z_train_30k) + len(z_valid_30k):]
    name = f'DBSCAN_{metric}_eps_{eps:02}'
    train_preds_30k[name] = train_clusters
    valid_preds_30k[name] = valid_clusters
    gil_preds_30k[name] = gil_clusters
    train_summary, train_df = get_cluster_stats(train_preds_30k, name, 'peptide', 'z_1', kf=False)
    valid_summary, valid_df = get_cluster_stats(valid_preds_30k, name, 'peptide', 'z_1', kf=False)
    gil_summary, gil_df = get_cluster_stats(gil_preds_30k, name, 'peptide', 'z_1', kf=False)
    train_summary.to_csv(f'{ODIR}train_summary_{name}.csv')
    train_df.to_csv(f'{ODIR}train_df_{name}.csv')
    valid_summary.to_csv(f'{ODIR}valid_summary_{name}.csv')
    valid_df.to_csv(f'{ODIR}valid_df_{name}.csv')
    gil_summary.to_csv(f'{ODIR}gil_summary_{name}.csv')
    gil_df.to_csv(f'{ODIR}gil_df_{name}.csv')

start =dt.now()
train_preds_30k = pd.read_csv('/home/projects/vaccine/people/yatwan/tclustr/data/preds_30k/train_preds_30k.csv')
valid_preds_30k = pd.read_csv('/home/projects/vaccine/people/yatwan/tclustr/data/preds_30k/valid_preds_30k.csv')
gil_preds_30k = pd.read_csv('/home/projects/vaccine/people/yatwan/tclustr/data/preds_30k/gil_preds_30k.csv')
latent_dim=64
z_train_30k = train_preds_30k[[f'z_{i}' for i in range(latent_dim)]].values
z_valid_30k = valid_preds_30k[[f'z_{i}' for i in range(latent_dim)]].values
z_gil_30k = gil_preds_30k[[f'z_{i}' for i in range(latent_dim)]].values

ODIR='/home/projects/vaccine/people/yatwan/tclustr/output/bruteforce_clustering/'
os.makedirs(ODIR, exist_ok=True)
ntm = partial(bruteforce, metric='cosine', ODIR=ODIR, train_preds_30k=train_preds_30k, valid_preds_30k=valid_preds_30k,
              gil_preds_30k=gil_preds_30k, z_gil_30k=z_gil_30k, z_train_30k=z_train_30k, z_valid_30k=z_valid_30k)
cosine_eps = np.linspace(0.25,1.5, 15)
Parallel(n_jobs=38)(delayed(ntm)(eps=eps) for eps in tqdm(cosine_eps, desc='n_clst', position=0, leave=True))

ntm = partial(bruteforce, metric='euclidean', ODIR=ODIR, train_preds_30k=train_preds_30k, valid_preds_30k=valid_preds_30k,
              gil_preds_30k=gil_preds_30k, z_gil_30k=z_gil_30k, z_train_30k=z_train_30k, z_valid_30k=z_valid_30k)
eucl_eps = np.linspace(8,14, 15)
Parallel(n_jobs=38)(delayed(ntm)(eps=eps) for eps in tqdm(eucl_eps, desc='n_clst', position=0, leave=True))



end = dt.now()
elapsed = divmod((end - start).seconds, 60)
print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')