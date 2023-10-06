import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os, sys, pickle
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import *


def get_cluster_stats(input_df, cluster='KMeans_Cluster', label='GroundTruth', feature='TSNE_1', kf=True):
    if kf:
        summaries = []
        dfs = []
        for fold in input_df.partition.unique():
            df = input_df.query('partition==@fold')
            l = len(df)
            df = df.groupby([cluster, label]).agg(count=(feature, 'count')).reset_index()
            summary = df.loc[df.groupby([cluster])['count'].idxmax(), [cluster, label]]
            cluster_sizes = df.groupby([cluster]).agg(cluster_size=('count','sum'))
            df = df.set_index([cluster, label])
            df['intra_cluster_percent'] = df['count'] / cluster_sizes['cluster_size'] * 100
            summary['purity_percent'] = summary.apply(lambda x: df.loc[x[cluster], x[label]]['intra_cluster_percent'], axis=1)
            summary = pd.merge(summary, cluster_sizes.reset_index(), left_on=[cluster], right_on=[cluster])
            df.reset_index([label], inplace=True)
            df['cluster_size'] = summary['cluster_size']
            df['cluster_label'] = summary[label]
            df.reset_index(inplace=True)
            df.set_index([cluster, label], inplace=True)
            summary['partition'] = fold
            summary['retention'] = (100 * summary['cluster_size']) / l
            df['partition']=fold
            summaries.append(summary)
            dfs.append(df)
        return pd.concat(summaries), pd.concat(dfs)
    else:
        df = input_df
        l = len(df)
        df = df.groupby([cluster, label]).agg(count=(feature, 'count')).reset_index()
        summary = df.loc[df.groupby([cluster])['count'].idxmax(), [cluster, label]]
        cluster_sizes = df.groupby([cluster]).agg(cluster_size=('count','sum'))
        df = df.set_index([cluster, label])
        df['intra_cluster_percent'] = df['count'] / cluster_sizes['cluster_size'] * 100
        summary['purity_percent'] = summary.apply(lambda x: df.loc[x[cluster], x[label]]['intra_cluster_percent'], axis=1)
        summary = pd.merge(summary, cluster_sizes.reset_index(), left_on=[cluster], right_on=[cluster])
        df.reset_index([label], inplace=True)
        df['cluster_size'] = summary['cluster_size']
        df['cluster_label'] = summary[label]
        df.reset_index(inplace=True)
        df.set_index([cluster, label], inplace=True)
        summary['partition'] = 'ALL'
        df['partition']= 'ALL'
        summary['retention'] = 100 * summary['cluster_size'] / l
        return summary, df

def pkl_dump(obj, filename, dirname=None):
    if dirname is not None:
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, filename)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print(f'{os.path.abspath(filename)} saved.')


OUTDIR = '/home/projects/vaccine/people/yatwan/tclustr/output/231006_latent_analysis'
DATADIR = '/home/projects/vaccine/people/yatwan/tclustr/data/231006_latent_dists/'
files = [f'{DATADIR}{x}' for x in os.listdir(DATADIR)]
os.makedirs(OUTDIR, exist_ok=True)


for linkage in ['average', 'complete', 'single']:
    for file in files:
        latent_kind = file.split('cat_')[1].split('.csv')[0]
        dist_kind = os.path.basename(file).split('.csv')[0].split('_cat')[0].replace('231006_','')
        savename = f'aggclstr_{linkage}_{latent_kind}_{dist_kind}'
        df = pd.read_csv(file, index_col=0)
        base_df = pd.read_csv(f'/home/projects/vaccine/people/yatwan/tclustr/data/preds_30k/231006_concat_no_dupes_preds_{latent_kind}.csv')
        input_matrix = df.drop(columns = ['labels', 'ids', 'set'])
        labels = df['labels'].values
        ids = df['ids'].values
        sets = df['set'].values        
        ag = AgglomerativeClustering(metric='precomputed', linkage=linkage, n_clusters=n_clusters, distance_threshold=None, compute_distances=True)
        pred_clusters = ag.fit_predict(input_matrix)
        ag_df = pd.DataFrame(np.stack([input_matrix.index, pred_clusters, labels, ids, sets], axis=1), columns = ['TRB_CDR3', 'pred_cluster', 'labels', 'ids', 'set'])
        ag_df = pd.merge(base_df.rename(columns={'seq_id':'ids'}), ag_df, left_on=['TRB_CDR3', 'labels', 'ids', 'set'], right_on = ['TRB_CDR3', 'labels', 'ids', 'set'])
        summary = get_cluster_stats(ag_df, cluster='pred_cluster', label='labels', feature='TRB_CDR3', kf=False)[0]
        pkl_dump(f'{OUTDIR}{savename}_model.pkl')
        ag_df.to_csv(f'{OUTDIR}{savename}_ag_df.csv')
        summary.to_csv(f'{OUTDIR}{savename}_summary_df.csv')
sys.exit(0)
