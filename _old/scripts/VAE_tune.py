from __future__ import print_function, division
#Allows relative imports
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#imports from files
from src.preprocessing import *
from src.VAE_train import *
from src.VAE_VJ_train import *
from src.vautoencoders import *
from src.loss_metrics import *
from src.pickling import *
from src.datasets import *
from src.torch_util import str_to_bool
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
    parser = argparse.ArgumentParser(description = 'Trains various VAE models and tunes them')
   #Optim & Train stuff
    parser.add_argument('-nb_epochs', type = int, default = 40, help= 'nb epochs, default = 40')
    parser.add_argument('-latent_dim',type = int, default = 50, help = 'latent dimension, default = 50')
    parser.add_argument('-outdir', type = str, help='where to save results, by default creates the folder "results/" in ./output/')
    parser.add_argument('-subset', type = float, default = 1, help = 'Fraction, whether to subsample the train/test/val set')
    parser.add_argument('-batchsize', type = int, default = 2**15, help = 'batchsize, default = 2**15')
    parser.add_argument('-lr', type = float, default = 6.67e-4, help = "LR, defualt = 6.67e-4")
    parser.add_argument('-adaptive', type=float, nargs = "+", default = (3, 0.96), help = 'adaptive lr, default = (3, 0.96)')
    parser.add_argument('-wd', type = float, default = 3e-6, help = 'weight decay, default = 3e-6' )
   #Loss stuff
    parser.add_argument('-how', type = str, default = 'MSE', help = 'AutoEncoder Loss, can be "MSE", "BCE", "HEL", by default is MSE')
    parser.add_argument('-alpha', type = float, default = 1, help = 'alpha, weight of encoder loss, default = 1')
    parser.add_argument('-beta', type = float, default = 7.5e-2, help = 'Beta, weight of KLD loss, default = 7.5e-2')
    parser.add_argument('-cyclic', type = str_to_bool, default = False, help = 'Cyclic behaviour for Beta (KLD loss)')
    parser.add_argument('-hyperbolic', type = str_to_bool, default =False, help = 'Tanh behaviour for Beta (KLD loss)')
    parser.add_argument('-gamma', type = float, default = 1, help='Gamma for Cyclic/Tanh ')
   #Encoding stuff
    parser.add_argument('-weighted', type = float, default = 1, help = 'Weight for encoding of first/last 4 amino acids')
    parser.add_argument('-positional', type = str_to_bool, default = True, help = 'Positional encoding behaviour')
    parser.add_argument('-pad', type = str, default = 'before', help = 'Pad before or after')
   #filename
    parser.add_argument('-name', type = str, default = 'VAE_tune', help ='filename')
    parser.add_argument('-seed', type = int, default = None, help = 'Manual seed')
    return parser.parse_args()
# Redo using EXP

def main():
    args = args_parser()
    start_time = dt.now()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
    print("\nARGS:",args,"\n")
    if not os.path.exists('../output/'):
        os.makedirs('../output/', exist_ok = True)  
        
    OUTDIR = os.path.join('../output/', args.outdir)
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
    
    #Reloading stuff 
    print('Loading data')
    train_dataset = load_pkl('train_dataset.pkl')
    valid_dataset = load_pkl('valid_dataset.pkl') 
    test_dataset = load_pkl('test_dataset.pkl') 
        
        
    g0 = math.ceil(len(train_dataset)/2**15) # = N batches, ie ~1 epoch
    if args.subset < 1 : 
        train_dataset = np.random.choice(train_dataset, math.ceil(args.subset * len(train_dataset)))
        valid_dataset = np.random.choice(valid_dataset, math.ceil(args.subset * len(valid_dataset)))
        test_dataset = np.random.choice(test_dataset, math.ceil(args.subset * len(test_dataset)))
    #print('g0', g0)
    #Feeding**kwargs to tune_vae
    d = vars(args)
    d['device'] = device
    d['outdir'] = OUTDIR
    #print('GAMMA HERE', d['gamma'])
    d['gamma'] *= g0
    #print('GAMMA THERE', d['gamma'])
    with open(os.path.join(OUTDIR, d['name']+'args.txt'), 'w') as f:
        f.write("###### ARGS::\n")
        for k in d.keys():
            f.write(f"{k}:{d[k]}\n")
    del d['seed']
    del d['subset']
    tune_vae(train_dataset, valid_dataset, test_dataset, **d)
    
    end_time = dt.now()       
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print("\nTime elapsed:\n\t{} minutes\n\t{} seconds".format(elapsed[0], elapsed[1]))

if __name__ == '__main__':
    main()
