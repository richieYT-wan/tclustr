from umap import UMAP
import seaborn as sns
import pandas as pd
import numpy as np 
import torch
from matplotlib import pyplot as plt
import os, glob
from tqdm.auto import tqdm
import colorcet as cc

def get_umap_and_plot_no_split(df, umap_kwargs={'metric':'euclidean', 'output_metric':'euclidean'}, palette=None,
                               figsize=(15,7), fold = 0, name='', filename='fn', order=None):
    
    z = df[[x for x in df.columns if x.startswith('z_')]].values
    labels = df['peptide'].values
    # TSNE extracting
    umap = UMAP(n_components=2, **umap_kwargs)
    latent_umap = umap.fit_transform(z)
    df[['UMAP_1', 'UMAP_2']] = latent_umap
    df['pred_label'] = df.apply(lambda x: x['label'] if x['purity_percent']>=66.67 else 'trash', axis=1)
    # plotting params
    pep_order = sorted(df.peptide.unique()) + ['trash'] if order is None else order + ['trash']
    palette = sns.color_palette(cc.glasbey, n_colors=len(pep_order)) if palette is None else sns.color_palette(palette, len(pep_order))
    sns.set_palette(palette)
    f,a = plt.subplots(1,2, figsize=figsize)
    a = a.ravel()
    # plotting
    sns.scatterplot(data=df, 
                    x='UMAP_1', y='UMAP_2',  s=13, hue='pred_label', hue_order=pep_order, ax = a[0])
    sns.scatterplot(data=df, 
                    x='UMAP_1', y='UMAP_2',  s=13, hue='peptide', hue_order=pep_order, ax = a[1])
    # labeling
    a[0].set_title('Train : Predicted clusters', fontsize=24, fontweight='semibold')
    a[1].set_title('Train : Ground Truth', fontsize=24, fontweight='semibold')
    a[0].legend('', frameon=False)
    a[1].legend(bbox_to_anchor=(1.5, 1))
    # Saving
    f.suptitle(f'Fold {fold} ; UMAP with AggloClusters on Cosine dist matrix, {name}', fontsize=26, fontweight='semibold', )
    # f.tight_layout()
    f.savefig(f'../output/240110_REDO_ClusteringComparisons/plots/umap/UMAP_Merged_{filename}.png', bbox_inches='tight', dpi=200)
    return df, f

df_best = pd.read_csv('../output/240110_REDO_ClusteringComparisons/df_best_e_0_t02112.csv')

# some loops to plot singles 
for n_neighbors in tqdm([3, 5, 8, 10, 15, 20, 25, 50], desc='n_neighbors '):
    for min_dist in tqdm([0.05, 0.1, 0.25, 0.4, 0.5, 0.7], desc='min_dist'):
        for learning_rate in tqdm([0.1, 0.5, 1.0, 1.2, 1.5, 2.5, 5], desc='learning_rate '):
            for spread in tqdm([0.5, 1.0, 1.2, 1.5], desc='spread '):
                for repulsion_strength in tqdm([0.1, 0.5, 1.0, 1.5], desc='repulsion_strength '):
                    for metric in ['cosine', 'euclidean']:
                        for output_metric in ['cosine', 'euclidean']:
                            umap_kwargs={'metric':metric, 'output_metric': output_metric,
                                         'min_dist':min_dist, 'n_neighbors':n_neighbors, 'spread':spread,
                                         'learning_rate': learning_rate, 'repulsion_strength': repulsion_strength}
                            fn = '-'.join([f'{k}_{v}' for k,v in umap_kwargs.items()])
                            if os.path.exists(f'../output/240110_REDO_ClusteringComparisons/plots/umap/UMAP_Merged_{fn}.png'):
                                continue
                            umap_kwargs['n_jobs']=2
                            df_plot_best, f_best = get_umap_and_plot_no_split(df_best, umap_kwargs=umap_kwargs,
                                                                     figsize=(15, 7), fold = 0, name='Normal VAE w/ FullTCR, best', filename=f'{fn}')