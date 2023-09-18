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
from vae_cel.vae_cel_loss import *
from vae_cel.vae_cel_train import tune_vae, resume_training

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
    parser = argparse.ArgumentParser(description = 'Trains various VAE models and tuxnes them')
    #parser.add_argument('-path', type = str, default = '../training_data_new/mixed_vj_dataset/', 
    #                    help= 'path to data')
   #Optim & Train stuff
    parser.add_argument('-pretrained', type = str_to_bool, default = False, help = 'If true, will reload weights (provided in -weightdir) and resume training from there')
    parser.add_argument('-weightpath', type = str, default = None, help = 'path to weights for pretrained model')
    parser.add_argument('-max_len', type = int, default = 23)
    parser.add_argument('-nb_epochs', type = int, default = 40, help= 'nb epochs, default = 40')
    parser.add_argument('-latent_dim',type = int, default = 50, help = 'latent dimension, default = 50')
    parser.add_argument('-outdir', type = str, help='where to save results, by default creates the folder "results/" in ./output/')
    parser.add_argument('-subset', type = float, default = 1, help = 'Fraction, whether to subsample the train/test/val set')
    parser.add_argument('-batch_size', type = int, default = 2**14, help = 'batch_size, default = 2**14')
    parser.add_argument('-lr', type = float, default = 6.67e-4, help = "LR, defualt = 6.67e-4")
    parser.add_argument('-adaptive', type=float, nargs = "+", default = (3, 0.96), help = 'adaptive lr, default = (3, 0.96)')
    parser.add_argument('-wd', type = float, default = 3e-6, help = 'weight decay, default = 3e-6' )
   #Loss stuff
    parser.add_argument('-how', type = str, default = 'CEL', help = 'deprecated must be removed (useless)')
    parser.add_argument('-alpha', type = float, default = 1, help = 'alpha, weight of encoder loss, default = 1')
    parser.add_argument('-beta', type = float, default = 7.5e-2, help = 'Beta, weight of KLD loss, default = 7.5e-2')
    parser.add_argument('-cyclic', type = str_to_bool, default = False, help = 'Cyclic behaviour for Beta (KLD loss)')
    parser.add_argument('-hyperbolic', type = str_to_bool, default =False, help = 'Tanh behaviour for Beta (KLD loss)')
    parser.add_argument('-linear', type = str_to_bool, default = False, help = 'linear behaviour for beta kld loss')
    parser.add_argument('-gamma', type = float, default = 1, help='Gamma for Cyclic/Tanh ')

   #Encoding stuff
    parser.add_argument('-weighted', type = float, default = 1, help = 'Weight for encoding of first/last 4 amino acids')
    parser.add_argument('-positional', type = str_to_bool, default = True, help = 'Positional encoding behaviour')
    parser.add_argument('-pad', type = str, default = 'before', help = 'Pad before or after')
   #filename
    parser.add_argument('-name', type = str, default = 'VAE_tune', help ='filename')
    parser.add_argument('-seed', type = int, default = None, help = 'Manual seed')
    parser.add_argument('-atchley', type = str_to_bool, default = False)
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
    train = pd.read_csv('../training_data_new/mixed_vj_dataset/mixed_vj_train.csv', usecols = ['amino_acid', 'v_family', 'j_family']).query('amino_acid.str.len() <= 23 and amino_acid.str.len() >=10')
    test = pd.read_csv('../training_data_new/mixed_vj_dataset/mixed_vj_test.csv', usecols = ['amino_acid', 'v_family', 'j_family']).query('amino_acid.str.len() <= 23 and amino_acid.str.len() >=10')
    
    if args.subset != 1 : 
        train = train.sample(frac = args.subset)
        test = test.sample(frac = args.subset)
        
    valid = train.sample(frac=0.25)
    train_dataset = train.drop(valid.index).values
    valid_dataset = valid.values
    test_dataset = test.values
    del train
    del valid
    
    g0 = len(train_dataset)/args.batch_size # = N batches, ie ~1 epoch
    
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
    del d['atchley']
    del d['max_len']
    if d['pretrained'] == False:
        del d['pretrained']
        del d['weightpath']
        tune_vae(train_dataset, valid_dataset, test_dataset, **d)
        
    elif d['pretrained'] == True:
        del d['pretrained']
        print(f'Reloading weights from {args.weightpath} and resuming training')
        if not d['weightpath'].endswith('.pth.tar'):
            
            print('Inferring best weights from directory.')
            d['weightpath'] = os.path.join(d['weightpath'],[x for x in os.listdir(d['weightpath']) if ('best' in x.lower() and x.endswith('.pth.tar'))][0])
            
        resume_training(train_dataset, valid_dataset, test_dataset, **d)
    
    end_time = dt.now()       
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print(f"\nTime elapsed:\n\t{elapsed[0]} minutes\n\t{elapsed[1]} seconds")
    
if __name__ == '__main__':
    main()
