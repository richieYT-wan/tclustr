"""Defines various losses and metrics used during training/prediction/classification"""

import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch.nn.functional as F
import torch.nn as nn 
import torch
from src.preprocessing import * 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

class VAELoss_cel(nn.Module):
    """
    Module defining a VAE loss combining either of the following 3 losses and a Kullback Leibler Divergence, weighted with alpha and beta respectively
    returned as : alpha * Reconstruction loss + beta * KLD
    how = 'HEL', 'MSE', 'BCE' --> HingeEmbeddingLoss, Mean Squared Error, Binary Cross Entropy
    Inherits from torch.nn.Module (so __call__ will call self.forward)
    
    Cyclic : If True, will implement sinusoid behaviour for Beta with a slight (0.3) shift:
    beta = beta_0 * (0.1+ [sin(-0.3 + (x/gamma) * pi)^2]/1.1 ) ; i.e. beta = beta_0 or 0 periodically
    
    Hyperbolic : If True, will implement tanh behaviour for Beta (converges towards beta_0)
    beta = beta_0 * 0.5 (1+ tanh( 0.001x - gamma));
    i.e. : beta = 0.5 * beta_0 at gamma * 1e3 and beta = beta_0 at ~2.1*gamma*1e3
    
    gamma gives either the period (1/frequency) for the sine or the shift for the hyperbolic (tanh) behaviour
    
    """
    def __init__(self, alpha=1, beta=1e-2, cyclic=False, hyperbolic = False, linear = False, gamma = None, atchley = False):
        super(VAELoss_cel, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # When using CEL, should transpose (1,2) on the x and do torch.argmax(x, dim=2)
        # But should also get rid of the positional encoding or separate it somehow

        self.alpha = alpha
        self.beta = beta
        self.cyclic = False
        self.hyperbolic = False
        self.linear = False
        self.gamma = 1
        self.atchley = atchley
        if cyclic == True:
            self.beta_0 = beta
            self.beta = 0
            self.cyclic = True
            self.count = 0
            self.gamma = gamma
            
        if hyperbolic == True:
            self.beta_0 = beta
            self.beta = 0
            self.hyperbolic = True
            self.count = 0
            self.gamma = gamma
            
        if linear == True : 
            self.beta_0 = beta
            self.beta = 0
            self.linear = True
            self.count = 0
            self.gamma = gamma #Gives slope = (1/gamma)
            
            
    def forward(self, x_hat, x, mu, logvar):
        """x and x should be tuples of (amino_acid, v, j) """
        if self.cyclic == True : 
            self.beta = self.beta_0*(0.1+ math.pow(math.sin((-0.3+ math.pi*(self.count/(self.gamma)))), 2)/1.1)
            #Only steps during training
            if self.training == True : 
                self.count += 1
        
        if self.hyperbolic == True :
            self.beta = self.beta_0 *0.5* (1+ math.tanh(0.2*(self.count - self.gamma))) 
            #Only steps during training
            if self.beta > (0.99995* self.beta_0):
                print('beta_0 reached')
                self.hyperbolic = False
                self.beta = self.beta_0
                              
            if self.training == True :
                self.count += 1
        
        if self.linear == True:
            self.beta = self.beta_0 * self.count/self.gamma
            if self.istrain == True:
                self.count += 1
        
        xh_aa, xh_pos, xh_v, xh_j, x_aa, x_pos, x_v, x_j = self.rearrange(x_hat,x)
        
        # Reconstruction loss, CEL loss with different weight to each of the elements.
        # aa reconstruction should have largest weight (1)
        aa_loss = self.criterion(xh_aa, x_aa.long())
        pos_loss = self.criterion(xh_pos, x_pos.long())
        v_loss = self.criterion(xh_v, x_v.long())
        j_loss = self.criterion(xh_j, x_j.long())
        
        REC = (1* aa_loss + 0.75 * pos_loss + 0.5 * v_loss + 0.33 * j_loss)/ (1+.75+.5+.33) # Weighted sum of the different losses
        
        KLD = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))# == /n_samples
        return self.alpha*REC + self.beta*KLD
    
    #already inherited from nn.Module
    # def train(self):
    #     self.istrain = True
    # def eval(self):
    #     self.istrain = False
    
    def rearrange(self, x_hat:tuple, x:tuple):
        """
        Splits tuples of x, x into their corresponding components,
        reshape and permutes them as needed, transforms from onehot to int encoding, etc.
        They should be tuples of content (onehot_aa, onehot_v, onehot_j)
        """
        #Splitting x
        x_hat_aa = x_hat[0].view(-1, x[0].shape[1], x[0].shape[2]) #corresponds to .view(-1, 23, 25)
        x_hat_pos = x_hat_aa[:,:,-4:].permute(0, 2, 1) #shape (N, 4, 23) as required by nn.CEL
        x_hat_aa = x_hat_aa[:,:,:21].permute(0, 2, 1) #shape (N, 21, 23) as required by nn.CEL
        x_hat_v = x_hat[1]# 2nd element of the tuple x, of shape(N,30)
        x_hat_j = x_hat[2]# 3rd element of the tuple x, of shape(N,2)
        
        #Splitting and taking argmax of x, same stuff
        oh_aa = x[0]#onehot_aa is the first item
        x_pos = torch.argmax(oh_aa[:,:,-4:], dim = 2)
        x_aa = torch.argmax(oh_aa[:,:,:21], dim = 2)
        x_v = torch.argmax(x[1], dim = 1)
        x_j = torch.argmax(x[2], dim = 1)
        
        return x_hat_aa, x_hat_pos, x_hat_v, x_hat_j, x_aa, x_pos, x_v, x_j
        

def compute_metrics(x_true, x_hat, seq_original, seq_reconstructed):
    """for vae_cel, takes tuples as input"""
    #splitting
    (x_aa, x_v, x_j) = x_hat
    (onehot, true_v, true_j) = x_true
    
    results = {}
    hamm_total, hamm_seq = HammingBatch(seq_original, seq_reconstructed)
    results['hamming_seq_padded'] = hamm_total
    results['hamming_sequence'] = hamm_seq
    #results['accuracy_seq'] = accuracy_score(onehot.flatten(), x_aa.flatten())
    results['accuracy_V'] = accuracy_score(true_v, x_v)
    results['accuracy_J'] = accuracy_score(true_j, x_j)
    return results

def HammingDistance(string1, string2, norm=True):
    """
    Computes the (normalized) hamming distance between two sequences of amino acids.
    Both sequences must be the same length and as type string
    When dealing with reconstruction, string1 should be the original sequence
    and string2 should be the reconstructed sequence
    """
    # Start with a distance of zero, and count up
    sequence_error = 0 #error within sequence (excludes padding X)
    distance = 0 #Total difference (including padding X)
    # Loop over the indices of the string
    L = len(string1)
    L2 = string1.find('X') #Gives the true length of the original sequence without padding
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
            if i < L2:
                sequence_error += 1 # Only compares up to the padding
                
    # Return the final count of differences
    if norm:
        normed = 100*distance/L
        seq_normed = 100*sequence_error/L2 #% of wrong amino acid within the true sequence without padding
        return normed, seq_normed
    else :
        return distance, sequence_error

def HammingBatch(seqs1, seqs2, norm=True):
    """Returns the mean (normalized) hamming distance of sequences in a batch"""
    hamm_total = []
    hamm_seq = []
    for s1,s2 in zip(seqs1, seqs2):
        total_error, sequence_error = HammingDistance(s1,s2, norm=norm)
        hamm_total.append(total_error)
        hamm_seq.append(sequence_error)
    return np.mean(hamm_total), np.mean(hamm_seq)
