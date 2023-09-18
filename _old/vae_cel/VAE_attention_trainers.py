from __future__ import print_function, division
from datetime import datetime as dt

#Allows relative imports
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#imports from files
from src.preprocessing import *
from src.pickling import *
from src.torch_util import *
from src.repertoire_dataset import EmersonRepertoire_Dataset, load_train_test_repertoire
from vae_cel.DeepRC_VAE import *
from vae_cel.vae_cel import *

import pandas as pd 
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from sklearn.metrics import roc_auc_score, f1_score
#checking gpu status
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using : {}".format(device))
else:
    device = torch.device('cpu')
    print("Using : {}".format(device))
    

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
    
def train_model(model, criterion, optimizer, dataset, train_subset, batch_size, 
                max_len, weighted, pad, positional, device):
    """trains for one full epoch (all batches)"""
    model.train()
    train_loss = 0
    for b in tqdm(BatchSampler(RandomSampler(train_subset), 
                          batch_size = batch_size, drop_last = False),
                  desc = 'Train Batch',
                  leave = False, position = 3):
        
        values, n_per_bag, target = dataset[b]
        target = torch.Tensor(target).to(device).float() # apparently its long for CEL ?? but float for BCE ? lol wtf pytorch
        x_tuple = batch_aa_vj(values, max_len, weighted, pad,
                              positional = True, atchley = False, device = device)
        
        output = model(x_tuple, n_per_bag)
        loss = criterion(output.flatten(), target.flatten())#here flatten because bceloss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= math.floor(len(train_subset)/ batch_size)
    return train_loss

def eval_model(model, criterion, dataset, val_subset, batch_size, 
                max_len, weighted, pad, positional, device):
    """trains for one full epoch (all batches)"""
    model.eval()
    val_loss = 0
    rocs = []
    f1s = []
    with torch.no_grad():
        for b in tqdm(BatchSampler(RandomSampler(val_subset), 
                              batch_size = batch_size, drop_last = False),
                      desc = 'Valid Batch',
                      leave = False, position=4):
            
            values, n_per_bag, target = dataset[b]
            target = torch.Tensor(target).to(device).float() # apparently its long for CEL ?? but float for BCE ? lol wtf pytorch 
            x_tuple = batch_aa_vj(values, max_len, weighted, pad,
                                  positional = True, atchley = False, device = device)
            output = model(x_tuple, n_per_bag) #score of positive class (After sigmoid)
            
            loss = criterion(output.flatten(), target.flatten())#here flatten because bceloss
            val_loss += loss.item()
            #y_pred = quantize(output, 0.51)
            #roc_auc = roc_auc_score(target.cpu().numpy(), output.detach().cpu().numpy()) #roc_auc_score(y_true, y_score)
            #f1 = f1_score(target.detach.cpu().numpy(), y_pred.numpy())
            #rocs.append(roc_auc)
            #f1s.append(f1)
            
        #tqdm.write(f'ROC AUC Score : {np.mean(rocs)}//\tF1 Score: {np.mean(f1s)}')
        
    val_loss /= math.floor(len(val_subset)/ batch_size)
    return val_loss

def train_attention(model, nb_epochs, lr, wd, batch_size,
                    train_dataset, train_subset, val_subset, 
                    max_len=23, weighted=0.5, pad = 'before',
                    positional = True, device = 'cuda', outdir = os.getcwd(), name = ''):
    
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr, wd = wd)
    train_losses = []
    val_losses = [] 
    best_val = 100
    
    model.to(device) 
    
    for e in tqdm(range(nb_epochs),
                  position=0, leave = False):
        train_loss = train_model(model,criterion, optimizer, train_dataset, train_subset,
                                 batch_size, max_len, weighted, pad, 
                                 positional, device)
        
        val_loss = eval_model(model, criterion, train_dataset, val_subset, batch_size, 
                              max_len, weighted, pad, positional, device)
        
        if e != 0 and val_loss < best_val:
            torch.save({'state_dict':model.state_dict(), 'epoch':e, 'loss':val_loss}, 
                           os.path.join(outdir,f'best_VAE_attention_{name}.pth.tar'))
            best_val = val_loss
            
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        tqdm.write(f' loss at {e} epochs: TRAIN:{train_loss:.3e},\t VAL:{val_loss:.3e}')
        
    losses = {'train': train_losses,
              'val' : val_losses}
    return losses

from clf.MLP_binary import *
def pipeline_attention(datapath, top_k, allele, pos_class, split_ratio, nb_epochs, lr, batch_size,
             max_len=23, weighted=0.5, pad = 'before', wd = 1e-7,
             positional = True, device = 'cuda', outdir = os.getcwd(), name = ''):
    
    #start_time = dt.now()
    print('Loading data, this may take a few minutes')
    train_dataset, test_dataset, train_subset, val_subset = load_train_test_repertoire(path = datapath,
                                                                                   max_len = max_len, top_k = top_k, allele = allele, 
                                                                                   pos_class = pos_class, split_ratio = split_ratio)
    #fixed model for now
    print('Loading model')
    VAE = VAE_cel(latent_dim = 100, aa_dim = 25)
    path = '../output/HyperbolicContinueTraining/LOWERBETA_DirectlyMax/'
    VAE = load_model(VAE, os.path.join(path, [x for x in os.listdir(path) if 'best' in x.lower()][0]))
    #attention = AttentionNetwork(n_input_features = 100, n_layers = 3, n_units = 50)
    attention = MLP_AnotherBinary(100)
    output_net = OutputNetwork(n_input_features = 100, n_output_features = 1, 
                               n_layers = 3, n_units = 100)
    
    model = DeepRC_VAE(VAE,
                       attention,
                       output_net)
    
    print('Starting training')
    losses = train_attention(model, nb_epochs, lr, wd, batch_size,
                    train_dataset, train_subset, val_subset, 
                    max_len, weighted, pad,
                    positional, device, outdir, name)
    
    #end_time = dt.now()       
    #elapsed = divmod((end_time-start_time).total_seconds(), 60)
    #print(f"\nTime elapsed:\n\t{elapsed[0]} minutes\n\t{elapsed[1]} seconds")
    return losses
