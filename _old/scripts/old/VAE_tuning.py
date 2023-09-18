from __future__ import print_function, division
#Allows relative imports
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#imports from files
from src.preprocessing import *
from src.VAE_train import *
from src.pickling import *
from src.vautoencoders import * 
from src.loss_metrics import AutoEncoderLoss, HammingDistance
from src.torch_util import str_to_bool
from datetime import datetime as dt
from tqdm import tqdm 

import pandas as pd 
import numpy as np
import math

import argparse
#Plot and stuff
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def args_parser():
    parser = argparse.ArgumentParser(description = 'Trains various VAE models and tunes them')
    parser.add_argument('-weighted', type = str_to_bool, default = False, help='Whether to also test weighted onehot encoding')
    parser.add_argument('-outdir', type = str, help='where to save results, by default creates the folder "results/" in ./output/')
    parser.add_argument('-bn', nargs = "+", type = str_to_bool, default = False , help='do batchnorm or not. (can be -bn True False -otherargs ... to do both)')
    parser.add_argument('-etas', nargs = "+", type = float, default = [3e-4], help = "number of learning rates to test")
    parser.add_argument('-latent', nargs= "+", type = int, default = [20, 30, 40], help = 'latent dimensions to test')
    parser.add_argument('-drop', nargs = "+", type = float, default = [0, 0.3],
                        help = "number of dropout values")
    parser.add_argument('-activations', nargs= "+", default = ['relu','selu'],#['elu','leaky','relu','selu'],
                        help = "string of activations to try")
    parser.add_argument('-batchsize', nargs = "+", type = int, default = 2**13)
    parser.add_argument('-epochs', type = int, default = 30)
    parser.add_argument('-trainsize', type = float, default = 0.5, help ='Which fraction of the naive.csv to take for training, default = 0.5')
    parser.add_argument('-adaptive', type=float, nargs = "+", default = None)
    parser.add_argument('-dataset', type = str, default = 'naive', help = 'Which dataset to use : [naive, emerson]')
    return parser.parse_args()
ACT = {'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 'relu': nn.ReLU(), 'selu': nn.SELU()}

def main():
    
    start_time = dt.now()
    args = args_parser()
    print("\nARGS:",args,"\n")
    if not os.path.exists('./output/'):
        os.makedirs('./output/', exist_ok = True)  
        
    OUTDIR = os.path.join('./output/', args.outdir)
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok = True) 
    
    #checking gpu status
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using : {}".format(device))
    else:
        device = torch.device('cpu')
        print("Using : {}".format(device))
        
    #DATA HERE 
    if args.dataset == 'naive':
        df = pd.read_csv('./training_data_new/db_TRB.csv', header=0)#, nrows=1e5)
        df = df.dropna(subset =['cdr3_TRB'])[['cdr3_TRB','TRBV','TRBJ']]
        train = df.sample(frac=args.trainsize)
        valid = train.sample(frac=0.25)
        train = train.loc[train.index.difference(valid.index)]
        
        train_dataset = train.loc[train.cdr3_TRB.str.len()<=23].cdr3_TRB.values
        valid_dataset = valid.loc[valid.cdr3_TRB.str.len()<=23].cdr3_TRB.values
        
    elif args.dataset == 'emerson':
        df = pd.read_csv('./training_data_new/emerson_raw/batch1/emerson_batch1_train.tsv', sep='\t', header=0)
        train = df.sample(frac=args.trainsize)
        valid = train.sample(frac=0.25)
        train = train.loc[train.index.difference(valid.index)]
        
        train_dataset = train.loc[train.amino_acid.str.len()<=23].amino_acid.values
        valid_dataset = valid.loc[valid.amino_acid.str.len()<=23].amino_acid.values
        

    ############OTHER STUFF HERE
    activations = [ACT[k] for k in args.activations if k in ACT.keys()] #Make sure mis-inputs aren't a problem
    bn = args.bn 
    etas = args.etas
    weight = args.weighted
    drops = args.drop
    latent = args.latent
    #Fixed stuff
    mini_batch_size = 2**args.batchsize[0]
    nb_epochs = args.epochs
    lr = args.etas[0]
    criterion = AutoEncoderLoss
    if args.adaptive is not None:
        adaptive = tuple(args.adaptive)
    else: adaptive = None
        
    for lat in tqdm(latent, desc='LATENT'):
        for act in tqdm(activations, desc='ACT'):    
            for p_drop in tqdm(drops, desc='P_DROP'):
                name = '_'.join(['VAE','latdim'+str(lat), str(act).split('()')[0], 
                                      str(p_drop), 'lr'+str(lr)])
                fn_basic = 'Basic'+name
                fn_normal = 'Deep'+name
                # Seems like Deep VAE doesn't bring much with the extra layer so will only do Basic for now 
                ## ================== Deep VAE ========================
                #model = VAE(latent_dim=lat, in_dim = 21*23, act=act,
                #            p_drop = p_drop, batchnorm=False)
                #model.to(device)
                #optimizer = optim.Adam(model.parameters(), lr = lr)
                #
                #train_eval(model, criterion, optimizer, train_dataset, valid_dataset,
                #           mini_batch_size, 23, False, device, lr, 
                #           nb_epochs, outdir=OUTDIR, filename=fn_normal, adaptive = adaptive)
                #model.to('cpu')
                #del model
                #torch.cuda.empty_cache()
                # ==================== Basic =========================
                model_basic = VAE_basic(latent_dim = lat, in_dim = 21*23, 
                                        act= act)
                model_basic = model_basic.to(device)
                optimizer = optim.Adam(model_basic.parameters(), lr = lr)
                
                
                train_eval(model_basic, criterion, optimizer, train_dataset, valid_dataset,
                           mini_batch_size, 23, False, device, lr, 
                           nb_epochs, outdir=OUTDIR, filename=fn_basic, adaptive = adaptive)
                del model_basic
                torch.cuda.empty_cache()
                
                
    end_time = dt.now()       
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print("\nTime elapsed:\n\t{} minutes\n\t{} seconds".format(elapsed[0], elapsed[1]))

if __name__ == '__main__':
    main()
