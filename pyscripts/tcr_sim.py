import pandas as pd 
import numpy as np 
import pickle

fdir='/home/projects/vaccine/people/yatwan/tclustr/output/'
tcr_sim = pd.read_csv(f'{fdir}230928_NetTCR_CDR3B_pep.txt_all_sim.out', header=None,
                      sep = ' ', names=['all', 'A', 'B', 'score'], comment='#')
tcr_sim = tcr_sim.query('all=="ALL"').drop(columns = ['all'])

adj_matrix = tcr_sim.drop_duplicates().pivot(index='A', columns='B', values='score').fillna(0)
adj_matrix_np = adj_matrix.to_numpy()
adj_matrix.to_csv(f'{fdir}230929_similarity_adj_matrix.txt')
with open(f'{fdir}230929_similarity_adj_matrix_numpy.pkl', 'wb') as f:
	pickle.dump(adj_matrix_np, f)