import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import binomtest
from joblib import Parallel, delayed
from functools import partial

from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import os
from scipy.stats import binom_test

from src.cluster_utils import agglo_single_threshold


### The model match the patterns:
# *IMMREP25OS_CSTRP_TCRcluster_results*
# *IMMREP25OS_NOTRP_TCRcluster_results*
# *IMMREP25TS_NOTRP_TCRcluster_results*
# *IMMREP25TS_CSTRP_TCRcluster_results*
# seed_0000_ratio_1.0_IMMREP25_Ratio1_Run000_KFold_NoKF_250307_0028_IMMREP25OS_CSTRP_TCRcluster_results
# seed_0019_ratio_1.0_IMMREP25_Ratio1_Run019_KFold_NoKF_250307_1048_IMMREP25OS_CSTRP_clusters_summary.csv

def get_seed(fn):
    return int(fn.split('/')[-2].split('_ratio')[0].split('seed_')[1])


def get_binom(k):
    try:
        return binomtest(int(k), n=100, p=0.5, alternative="greater").pvalue
    except:
        return np.nan


def get_summary(output_df):
    summary = []
    for pred_label in sorted(output_df.cluster_label.unique()):
        tmp = output_df.query('cluster_label==@pred_label')
        tmp = tmp.groupby('label').agg(count=('B3', 'count'))
        tmp['perc'] = tmp['count'] / tmp['count'].sum()
        summary.append({'pred_label': pred_label,
                        'majority_label': tmp['perc'].idxmax(),
                        'purity': tmp.loc[tmp['perc'].idxmax()]['perc'],
                        'cluster_size': tmp['count'].sum()})
    return pd.DataFrame(summary)


def filter_single(seed_number, model_name, purity_threshold=0.75, size_threshold=5):
    out_file = glob(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_runs/*seed_{seed_number:04}*/*IMMREP25{model_name}_TCRcluster_results*.csv')
    # THE SUMMARY FILES ARE SOMEHOW WRONG SO RE-GET IT MANUALLY HERE
    # summary_file = glob(
    #     f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_runs/*seed_{seed_number:04}*/*IMMREP25{model_name}_clusters_summary*.csv')
    # assert (len(out_file) == 1 and len(summary_file) == 1), f'Couldn\'t find files for seed {seed_number}'
    out_df = pd.read_csv(out_file[0])
    sum_df = get_summary(out_df)
    # sum_df = pd.read_csv(summary_file[0])
    # Check the summaries
    sum_df = sum_df.query('majority_label=="sample" and purity>@purity_threshold and cluster_size>@size_threshold')
    queried_labels = sum_df['pred_label'].unique()
    # Remove the healthy backgrounds from the out_df and take only the matched outputs
    drop_columns = [x for x in out_df.columns if x.startswith('z_') or 'reconstructed' in x]
    out_df = out_df.query('cluster_label in @queried_labels and label=="sample"').drop(columns=drop_columns).assign(
        seed=seed_number)
    return out_df


def fill_adjacency(df, adj_matrix):
    for label in df.cluster_label.unique():
        tmp = df.query('cluster_label==@label')
        indices = tmp['index_col'].values
        adj_matrix.loc[indices, indices] += 1


def do_single_adj(adj_matrix, seed_number, model_name, purity_threshold=.75, size_threshold=5):
    df = filter_single(seed_number, model_name, purity_threshold, size_threshold)
    fill_adjacency(df, adj_matrix)
    return adj_matrix.values


def get_adj_dist_matrices(raw_df, purity_threshold, size_threshold, n_jobs=8):
    # WORKAROUND TO USE PARALLEL to fill adj matrix because otherwise it doesn't update the df
    adj_matrix = pd.DataFrame(columns=raw_df['index_col'].values, index=raw_df['index_col'].values).fillna(0)
    wrapper = partial(do_single_adj, adj_matrix=adj_matrix, model_name='OS_NOTRP', purity_threshold=purity_threshold,
                      size_threshold=size_threshold)
    adjs = Parallel(n_jobs=n_jobs)(delayed(wrapper)(seed_number=s) for s in tqdm(range(100), desc='fill adj'))
    adj_values = sum([adj for adj in adjs])
    # Fill diag with 100 (adjacency to self is 100 since it's always found with itself)
    np.fill_diagonal(adj_values, 100)
    adj_matrix.loc[adj_matrix.index, adj_matrix.index] = adj_values
    # Get the dist matrix (1 - X and normalise and fill diag with 0s)
    dist_matrix = adj_matrix.copy()
    dm_values = dist_matrix.values
    dm_values = 1 - (dm_values / 100)  # normalise distance to range (0,1)
    # Distance to self is 0
    np.fill_diagonal(dm_values, 0)
    dist_matrix.loc[dist_matrix.index, dist_matrix.index] = dm_values
    return adj_matrix, dist_matrix


def generate_all(dist_matrix):
    # Get random labels
    dist_matrix['fake_labels'] = [f'random_label_{i}' for i in np.random.randint(0, 5, len(dist_matrix))]
    dist_array = dist_matrix.iloc[:len(dist_matrix), :len(dist_matrix)].values
    labels = dist_matrix['fake_labels'].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return dist_matrix, dist_array, labels, encoded_labels, label_encoder


def rerun_clustering(raw_df, dist_matrix, threshold=None, pval_threshold=0.05):
    # convoluted cause I can't be bothered to rethink this it takes 1ms to compute
    threshold_pval_dict = {(100 - i) / 100: binom_test(i, 100) for i in range(50, 100)}
    pval_threshold_dict = {v: k for k, v in threshold_pval_dict.items()}
    pv_thresholds = pd.DataFrame({v: k for k, v in threshold_pval_dict.items()}, index=['dist']).T.reset_index().rename(
        columns={'index': 'pv'})

    dist_matrix, dist_array, labels, encoded_labels, label_encoder = generate_all(dist_matrix)
    # manual threshold overrides pval threshold
    if threshold is None:
        threshold = pv_thresholds.query('pv<@pval_threshold').iloc[0]['dist']
    metrics, clusters_df, c = agglo_single_threshold(dist_array, dist_array, labels, encoded_labels,
                                                     label_encoder, threshold,
                                                     return_df_and_c=True)
    # Remerge to input df
    dist_matrix.drop(columns=['fake_labels'], inplace=True)
    dist_matrix['pred_label'] = c.labels_
    merged_output = pd.merge(raw_df.set_index('index_col'), dist_matrix['pred_label'], left_index=True,
                             right_index=True)

    merged_output = merged_output.merge(
        merged_output.groupby(['pred_label']).agg(cluster_size=('B3', 'count')).reset_index(),
        left_on=['pred_label'], right_on=['pred_label'])
    merged_output['cluster_threshold'] = threshold
    return merged_output


def pipeline(model_name, purity_threshold=0.75, size_threshold=6):
    # Get the path
    # dirpath = glob(f'../output/2503XX_IMMREP25_output/score_vs_healthy/*seed_{seed_number:04}*/')[0]
    # if not os.path.exists(dirpath):
    #     raise ValueError(f'Path not found for seed = {seed_number}; globbed dirpath :{dirpath}')
    # Read the files corresponding to a given number
    raw_file = '../data/IMMREP25/test.csv_fmt4TCRcluster_uniq'
    raw_df = pd.read_csv(raw_file)
    raw_df['index_col'] = [f'sample_{i:05}' for i in range(len(raw_df))]

    # debug :
    filtered = []
    # for s in range(100):
    #     filtered.append(filter_single(s, model_name, purity_threshold, size_threshold))
    # filtered = pd.concat(filtered)
    filter_wrapper = partial(filter_single, model_name=model_name, purity_threshold=purity_threshold,
                             size_threshold=size_threshold)
    filtered = pd.concat(Parallel(n_jobs=8)(delayed(filter_wrapper)(seed_number=s) for s in tqdm(range(100), desc='filter seeds')))

    # Get the occurences (out of 100)
    counts = filtered.groupby(['index_col']).agg(cluster_count=('seed', 'count'))
    counts['cluster_percent'] = counts['cluster_count'] / 100
    # Merge to the input df
    merged_df = raw_df.set_index(['index_col']).merge(counts, left_index=True, right_index=True, how='left')
    # merged_df['cluster_count'] = merged_df['cluster_count'].astype(int)
    merged_df['cluster_binom_pval'] = merged_df['cluster_count'].apply(get_binom)
    merged_df['cluster_binom_sig'] = merged_df['cluster_binom_pval'].apply(
        lambda x: 'ns' if x > 0.05 else '****' if x <= 0.0001 else '***' if x <= 0.001 else '**' if x <= 0.01 else '*')

    merged_df.to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/all_counts_binom_pvals.txt')
    merged_df.query('cluster_binom_sig=="ns"').to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_siglvl_ns.csv')
    merged_df.query('cluster_binom_sig=="*"').to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_siglvl_1.csv')
    merged_df.query('cluster_binom_sig=="**"').to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_siglvl_2.csv')
    merged_df.query('cluster_binom_sig=="***"').to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_siglvl_3.csv')
    merged_df.query('cluster_binom_sig=="****"').to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_siglvl_4.csv')
    sorted_df = merged_df.sort_values('cluster_count', ascending=False)
    sorted_df.head(len(merged_df) // 100).to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_top1_percent.csv')
    sorted_df.tail(len(merged_df) // 100).to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_bot1_percent.csv')
    # top50 / bot50
    sorted_df.head(50).to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_top50.csv')
    sorted_df.tail(50).to_csv(
        f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/concat_bot50.csv')

    # Here, from the raw_file, re-run the clusters:
    adj_matrix, dist_matrix = get_adj_dist_matrices(raw_df, purity_threshold, size_threshold, -1)
    for pv in [0.05, 0.01, 0.001, 0.0001]:
        wtf = f'{pv:e}'
        wtf = wtf.split('.')[0] + 'e' + wtf.split('e')[1]
        cluster_output = rerun_clustering(raw_df, dist_matrix, None, pv)
        cluster_output.to_csv(f'../output/2503XX_IMMREP25_output/score_vs_healthy/rerun_cluster_{model_name}_pv_{wtf}.csv')


model_names = ['OS_NOTRP', 'OS_CSTRP', 'TS_NOTRP', 'TS_CSTRP']

# TODO: Need to find something that finds the frequently co-clustered members into clusters together
for model_name in tqdm(model_names, desc='model_name'):
    print(model_name)
    os.makedirs(f'../output/2503XX_IMMREP25_output/score_vs_healthy/100_seeds_analysis/{model_name}/', exist_ok=True)
    pipeline(model_name, .75, 5)
