"""Redoes the VAE from scratch to accomodate how I want to use CEL"""

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from src.preprocessing import * 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#Plot and stuff
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')

from src.preprocessing import * 
#from src.loss_metrics import *
from src.pickling import * 
#from src.vautoencoders import *
from tqdm.auto import tqdm 
from torch.utils.data import BatchSampler, RandomSampler
#from src.datasets import * 

########################## VAE CLASS HERE 
class VAE_cel(nn.Module):
    """
    Input : Flattened&concatenated AA_onehot, v_onehot, j_onehot 
    shape (N, aa_dim*max_len + v_dim + j_dim), should be (N, 25*23+30+2) when using Positional encoding
    """
    def __init__(self, seq_len = 23, aa_dim = 21, 
                 latent_dim= 100, act = nn.SELU(), 
                 v_dim = 30, j_dim = 2):
        
        super(VAE_cel, self).__init__()
        self.seq_len = seq_len # = 23
        self.aa_dim = aa_dim # = 21
        self.in_dim = seq_len * aa_dim + v_dim + j_dim #23*25 + 32 = 607 with positional
        self.lat_dim = latent_dim # = 50
        self.v_dim = v_dim # = 30
        self.j_dim = j_dim # = 2
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), 100),
                                     act,
                                    )
        
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_var = nn.Linear(100, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(self.lat_dim,  100),
                                     act,
                                     nn.Linear(100,  100),
                                     act
                                     )
        
        self.out_aa = nn.Sequential(nn.Linear(100,math.floor(self.in_dim/2)),
                                    act,
                                    nn.Linear(math.floor(self.in_dim/2), 
                                              self.seq_len * self.aa_dim)
                                    )
            
        self.out_v = nn.Linear(100, self.v_dim) #these are logits for the class : 100-> 30
        self.out_j = nn.Linear(100, self.j_dim) #these are logits for the class : 100-> 2 
    
    def get_infos(self):
        tmp = {}
        tmp['max_len'] = self.seq_len
        tmp['aa_dim'] = self.aa_dim
        tmp['latent_dim']= self.lat_dim
        tmp['in_dim'] = self.in_dim
        
        return tmp
    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.empty_like(mu).normal_(mean=0,std=1) 
        #if self.training == True : 
        z = (epsilon*std) + mu
        #else:
        #    z = std + mu
        return z
    
    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1, self.in_dim))
        mu = self.fc_mu(mu_logvar)
        logvar = self.fc_var(mu_logvar)
        return mu, logvar
    
    def embed(self, x) :
        x = self.reshape_input_tuple(x)
        mu, logvar = self.encode(x)
        return self.reparameterise(mu, logvar) #returns z
    
    
    def decode(self, z):
        decoded = self.decoder(z)
        #I can do this, or maybe just do the ReLU thing for the aa/pos reconstruction (like normal VAE)
        #then another layers for V and J
        v_reconstructed = self.out_v(decoded) #Logits with no activation to use with CrossEntropyLoss
        j_reconstructed = self.out_j(decoded) #Logits with no activation to use with CrossEntropyLoss
        aa_reconstructed = self.out_aa(decoded) #This should contain the aa encoding AND the positional encoding of dimension (N, 23*25), ie. a flattened (dim1) tensor
        
        xs_hat = (aa_reconstructed, v_reconstructed, j_reconstructed)
        return xs_hat
    
    def reshape_input_tuple(self, x:tuple) :
        """flattens and concat the input which is a tuple of tensors"""
        x = [torch.flatten(tmp, start_dim=1) for tmp in x] # Flattens the vectors 
        x = torch.cat(x, dim = 1) #Concatenate them along dim1
        # --> returns a flat tensor of shape [batchsize, aa_dim*max_len + v_dim + j_dim]
        return x 
    
    def forward(self, x:tuple):
        """x should be a tuple containing the 3 one-hot (not flattened) vectors"""
        x = self.reshape_input_tuple(x)
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        xs_hat = self.decode(z) #this is a tuple, so its xs_hat instead of x
        return xs_hat, mu, logvar
    
    def embed_reconstruct(self, x):
        xs_hat, _, _ = self.forward(x)
        
        x_aa = xs_hat[0]
        x_v = xs_hat[1]
        x_j = xs_hat[2]
        
        return x_aa, x_v, x_j
        
        
    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.lat_dim)).to(device=self.encoder[0].weight.device)
        return z #returns a tuple
    
    def reconstruct_latent(self, z):
        x_tuple = self.decode(z)
        x_aa, x_v, x_j = x_tuple
        #Softmaxing and getting the V/J
        x_v = list(torch.argmax(F.softmax(x_v, dim = 1), dim = 1).detach().cpu().numpy())
        x_j = list(torch.argmax(F.softmax(x_j, dim = 1), dim = 1).detach().cpu().numpy())
        #reshaping and slicing aa vector
        x_aa = x_aa.view(-1, self.seq_len, self.aa_dim)[:,:,:21] 
        decoded = torch.argmax(F.softmax(x_aa, dim = 2), dim = 2)
        decoded = F.one_hot(decoded)
        sequences = decode_batch(decoded.view(-1, self.seq_len, 21))
        
        return decoded, sequences, x_v, x_j
        
        