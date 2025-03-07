import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import binomtest
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
import os
### The model names are :
# cluster_results_TBCRalign_agglo.csv
# cluster_results_VAE_OS_CsTRP_agglo.csv
# cluster_results_VAE_OS_NoTRP_agglo.csv
# cluster_results_VAE_TS_CsTRP_agglo.csv
# cluster_results_VAE_TS_NoTRP_agglo.csv
# cluster_results_tcrdist3_agglo.csv

def get_seed(fn):
    return fn.split('/')[-2].split('_MST')[0].split('seed_')[1]

def get_binom(k):
	try:
		return binomtest(int(k), n=100, p=0.5, alternative="greater").pvalue
	except:
		return np.nan

model_names = ['TBCRalign', 'VAE_OS_NoTRP', 'VAE_OS_CsTRP', 'VAE_TS_NoTRP', 'VAE_TS_CsTRP','tcrdist3']
def pipeline(francis_number, model_name, purity_threshold=0.8, size_threshold=5):
	# Read the files corresponding to a given number
	raw_file = glob(f'../data/OTS/francis_covid_042/*{francis_number:04}*.txt')[0]
	out_files = glob(f'../output/241002_subsampled_francis_garner_lowcount/*francis_{francis_number:03}_seed_*/*cluster_results_{model_name}_agglo*.csv')
	raw_df = pd.read_csv(raw_file)
	out_dfs = pd.concat([pd.read_csv(x).assign(seed=get_seed(x)) for x in out_files])
	# Filter the output cluster dfs by the size and purity criteria
	filtered = out_dfs.query('cluster_size>=@size_threshold and purity>=@purity_threshold and majority_label=="covid"')
	# Get the occurences (out of 100)
	counts = filtered.groupby(['index_col']).agg(cluster_count=('seed','count'))
	counts['cluster_percent'] = counts['cluster_count'] / 100
	# Merge to the input df
	merged_df = raw_df.set_index(['index_col']).merge(counts, left_index=True, right_index=True, how='left')
	# merged_df['cluster_count'] = merged_df['cluster_count'].astype(int)
	merged_df['cluster_binom_pval']=merged_df['cluster_count'].apply(get_binom)
	merged_df['cluster_binom_sig']=merged_df['cluster_binom_pval'].apply(lambda x: 'ns' if x > 0.05 else '****' if x <= 0.0001 else '***' if x <= 0.001 else '**' if x <= 0.01 else '*')

	merged_df.to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/francis_{francis_number:04}_counts_binom_pvals.txt')
	return merged_df.assign(file_number=francis_number)

# TODO : 
# For a given repertoire number (francis_XXX), read the file in ../data/francis_covid_042/
# There you can match the index_col 
# Get all 100 files (seeds) for a given run and a given model (files named cluster_results_MODELNAME_agglo.csv')
# df = df.query('majority_label=="covid" and purity>P_threshold and cluster_size>S_threshold')
# df.index_col --> count the occurences of francis_XXXX reappearing often out of 100 runs 
# could maybe get a binomial p-value for each sequence index, then log the sequences that have a certain significance level, 
number_range = range(43)
for model_name in tqdm(model_names, desc='model_name'):
	print(model_name)
	os.makedirs(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/', exist_ok=True)
	wrapper = partial(pipeline, model_name=model_name, purity_threshold=0.8, size_threshold=5)
	out = Parallel(n_jobs=-1)(delayed(wrapper)(francis_number=n) for n in tqdm(number_range))
	dfs = pd.concat(out).reset_index()
	dfs['merged_index'] = dfs.apply(lambda x: f"{x['file_number']:03}_{x['index_col']}", axis=1)
	dfs.query('cluster_binom_sig=="ns"').to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_siglvl_ns.csv')
	dfs.query('cluster_binom_sig=="*"').to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_siglvl_1.csv')
	dfs.query('cluster_binom_sig=="**"').to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_siglvl_2.csv')
	dfs.query('cluster_binom_sig=="***"').to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_siglvl_3.csv')
	dfs.query('cluster_binom_sig=="****"').to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_siglvl_4.csv')
	# top 1% / bot 1% 
	sorted_dfs = dfs.sort_values('cluster_count', ascending=False)
	sorted_dfs.head(len(dfs)//100).to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_top1_percent.csv')
	sorted_dfs.tail(len(dfs)//100).to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_bot1_percent.csv')
	# top50 / bot50
	sorted_dfs.head(50).to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_top50.csv')
	sorted_dfs.tail(50).to_csv(f'../output/250219_FrancisCovid_IndividualAnalyses/{model_name}/concat_bot50.csv')

