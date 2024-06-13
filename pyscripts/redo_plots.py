import os
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import glob
from joblib import Parallel, delayed

def redo_plot(file, identifier, dataset, folder):
	xx = pd.read_csv(file)
	xx['input_type']=xx['input_type'].apply(lambda x: x.replace(identifier, '').lstrip('_').rstrip('__'))
	# plotting options
	sns.set_style('darkgrid')
	sns.set_palette('gnuplot2', n_colors=len(xx.input_type.unique()) - 2)
	f, a = plt.subplots(1, 1, figsize=(9, 9))
	a.set_xlim([0, 1])
	a.set_ylim([0, 1])
	a.set_xlabel('Retention', fontweight='semibold', fontsize=14)
	a.set_ylabel('Avg Purity', fontweight='semibold', fontsize=14)
	# Setting major ticks
	major_ticks = np.arange(0, 1.1, 0.1)
	a.set_xticks(major_ticks)
	a.set_yticks(major_ticks)
	# Setting minor ticks
	minor_ticks = np.arange(0, 1.1, 0.05)
	a.set_xticks(minor_ticks, minor=True)
	a.set_yticks(minor_ticks, minor=True)
	plt.grid(which='both', linestyle='--', linewidth=0.5)

	for input_type in xx.input_type.unique():
	    query = xx.query('input_type==@input_type')
	    retentions = query['retention'][1:-1].values
	    purities = query['mean_purity'][1:-1].values
	    if input_type == "TBCRalign":
	        a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='g', lw=1.)
	    elif input_type == "tcrdist3":
	        a.plot(retentions, purities, label=input_type.lstrip('_'), ls='-.', c='y', lw=1.)
	    else:
	        a.plot(retentions, purities, label=input_type, ls='--', lw=1.)

	a.axhline(0.6, label='60% purity cut-off', ls=':', lw=.75, c='m')
	a.axhline(0.7, label='70% purity cut-off', ls=':', lw=.75, c='c')
	a.axhline(0.8, label='80% purity cut-off', ls=':', lw=.75, c='y')

	a.legend(title='distance matrix', title_fontproperties={'size': 14, 'weight': 'semibold'},
	         prop={'weight': 'semibold', 'size': 12})
	f.suptitle(f'{dataset} :: {identifier}', fontweight='semibold', fontsize=15)
	f.tight_layout()
	f.savefig(f'{folder}REDO_{dataset}_PurRetCurves_{identifier}.png', dpi=200)

train_files = glob.glob('../output/240516_TripletTweaks_IntervalClustering/*/*/*train*.csv')
valid_files = glob.glob('../output/240516_TripletTweaks_IntervalClustering/*/*/*valid*.csv')

train_ids = [os.path.basename(x).split('__')[0] for x in train_files]
valid_ids = [os.path.basename(x).split('__')[0] for x in valid_files]

train_folders = ['/'.join(x.split('/')[:-1])+'/' for x in train_files]
valid_folders = ['/'.join(x.split('/')[:-1])+'/' for x in valid_files]

Parallel(n_jobs=8)(delayed(redo_plot)(file=f, identifier=idf, dataset='train', folder=folder) for (f, idf, folder) in tqdm(zip(train_files, train_ids, train_folders), desc='train_folders'))
Parallel(n_jobs=8)(delayed(redo_plot)(file=f, identifier=idf, dataset='valid', folder=folder) for (f, idf, folder) in tqdm(zip(valid_files, valid_ids, valid_folders), desc='valid_folders'))
