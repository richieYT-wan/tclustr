from __future__ import print_function, division
#Allows relative imports
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#imports from files
from src.preprocessing import *
from src.pickling import *
from src.datasets import *
from src.torch_util import str_to_bool
from vae_cel.vae_cel import *
from vae_cel.VAE_attention_trainers import * 
from vae_cel.DeepRC_VAE import * 

import argparse
import pandas as pd 
import numpy as np
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime as dt

#Plot and stuff
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')

torch.cuda.empty_cache()
# Ignore warnings)
import warnings
warnings.filterwarnings("ignore")

def args_parser():
    parser = argparse.ArgumentParser(description = 'train VAE_attention model')
    parser.add_argument('-datapath', type = str, default = '../training_data_new/emerson_raw/batch1/', help = 'path to weights for datasets')
    parser.add_argument('-top_k', type = int, default = 10000, help ='top_k seq to take')
    parser.add_argument('-max_len', type = int, default = 23)
    parser.add_argument('-allele', type = str, default = 'A', help = 'which allele (A or B)')
    parser.add_argument('-pos_class', type = str, default = 'A01', help = 'which class as positive for OvR (default = A01)')
    parser.add_argument('-split_ratio', type = float, default = .7, help = "ratio in train set")
    parser.add_argument('-nb_epochs', type = int, default = 30, help= 'nb epochs, default = 40')
    parser.add_argument('-lr', type = float, default = 1e-3, help = "learning rate, def = 1e-3")
    parser.add_argument('-wd', type = float, default = 1e-9, help = "wd, def = 1e-9 i.e. almost none")
    parser.add_argument('-batch_size', type = int, default = 4, help = 'batchsize, default = 2**14')
    parser.add_argument('-weighted', type = float, default = .5, help = 'Weight for encoding of first/last 4 amino acids')
    parser.add_argument('-pad', type = str, default = 'before', help = 'Pad before or after')
    parser.add_argument('-positional', type = str_to_bool, default = True, help = 'Positional encoding behaviour')
    parser.add_argument('-outdir', type = str, default = 'Allele_Name_here' ,help='where to save results')
    parser.add_argument('-name', type = str, default = 'allele_name_here', help ='filename')
    parser.add_argument('-seed', type = int, default = 20, help = 'Manual seed, default = 20')
    
    return parser.parse_args()

def main():
    args = args_parser()
    start_time = dt.now()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
    print("\nARGS:",args,"\n")
    if not os.path.exists('../output/VAE_attention/'):
        os.makedirs('../output/VAE_attention/', exist_ok = True)  
        
    OUTDIR = os.path.join('../output/VAE_attention/', args.outdir)
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok = True) 
    print(f'Files will be saved at {OUTDIR}')
    #checking gpu status
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))
    
    d = vars(args)
    d['device'] = device
    d['outdir'] = OUTDIR
    #print(OUTDIR)
    with open(os.path.join(OUTDIR, d['name']+'args.txt'), 'w') as f:
        f.write("###### ARGS::\n")
        for k in d.keys():
            f.write(f"{k}:{d[k]}\n")
    del d['seed']
    losses = pipeline_attention(**d)
    
    for k in losses.keys():
        plt.plot(losses[k], label = k)
    plt.title(f'Loss during training for {d["pos_class"]}; {name}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(OUTDIR+f'{name}_losses.jpg')
    
    end_time = dt.now()       
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print(f"\nTime elapsed:\n\t{elapsed[0]} minutes\n\t{elapsed[1]} seconds")
    
if __name__ == '__main__':
    main()
    