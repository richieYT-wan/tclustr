import pickle
import os

#Small helper file
PATH = os.getcwd()
#Merged dict : [atchley1 ... atchley5, PCA1,...,PCA15]

"""wrapper functions for pickling"""
def save_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pkl(filename):
    with open(filename, 'rb') as f :
        obj = pickle.load(f)
    return obj

def load_dict(PATH):
    if 'notebook' in PATH or 'script' in PATH:
        hla_a = load_pkl('../src/pickles/hla_a.pkl')
        hla_b = load_pkl('../src/pickles/hla_b.pkl')
        minmax_atchley = load_pkl('../src/pickles/minmax_atchley.pkl')
    
    elif 'src' in PATH:

        hla_a = load_pkl('./pickles/hla_a.pkl')
        hla_b = load_pkl('./pickles/hla_b.pkl')
        minmax_atchley = load_pkl('./pickles/minmax_atchley.pkl')
        
    else :
        hla_a = load_pkl('./src/pickles/hla_a.pkl')
        hla_b = load_pkl('./src/pickles/hla_b.pkl')
        minmax_atchley = load_pkl('./src/pickles/minmax_atchley.pkl')
    return hla_a,  hla_b, minmax_atchley
        
def load_all(PATH):
    if 'notebook' in PATH:
        AAidx_Dict = load_pkl('../src/pickles/AAidx_dict.pkl')
        merged_dict = load_pkl('../src/pickles/merged_dict.pkl')
        minmax_aaidx = load_pkl('../src/pickles/minmax_aaidx.pkl')
        minmax_merged = load_pkl('../src/pickles/minmax_merged.pkl')
        minmax_atchley = load_pkl('../src/pickles/minmax_atchley.pkl')
        hla_a = load_pkl('../src/pickles/hla_a.pkl')
        hla_b = load_pkl('../src/pickles/hla_b.pkl')
        
    else :
        AAidx_Dict = load_pkl('./src/pickles/AAidx_dict.pkl')
        merged_dict = load_pkl('./src/pickles/merged_dict.pkl')
        minmax_aaidx = load_pkl('./src/pickles/minmax_aaidx.pkl')
        minmax_merged = load_pkl('./src/pickles/minmax_merged.pkl')
        minmax_atchley = load_pkl('./src/pickles/minmax_atchley.pkl')
        hla_a = load_pkl('./src/pickles/hla_a.pkl')
        hla_b = load_pkl('./src/pickles/hla_b.pkl')
    return AAidx_Dict, merged_dict , minmax_aaidx, minmax_merged, minmax_atchley,  hla_a,  hla_b
