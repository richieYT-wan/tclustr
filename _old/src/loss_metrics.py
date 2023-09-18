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


dict_ = {'HEL' : nn.HingeEmbeddingLoss(reduction = 'mean'),
         'MSE' : nn.MSELoss(reduction = 'mean'),
         'BCE' : nn.BCELoss(reduction = 'mean'),
         'CEL' : nn.CrossEntropyLoss()
        }


class VAELoss(nn.Module):
    """
    Module defining a VAE loss combining either of the following 3 losses and a Kullback Leibler Divergence, weighted with alpha and beta respectively
    returned as : alpha * Reconstruction loss + beta * KLD
    how = 'HEL', 'MSE', 'BCE' --> HingeEmbeddingLoss, Mean Squared Error, Binary Cross Entropy
    Inherits from torch.nn.Module (so __call__ will call self.forward)
    
    Cyclic : If True, will implement sinusoid behaviour for Beta with a slight (0.3) shift:
    beta = beta_0 * sin(-0.3 + (x/gamma) * pi)^2 ; i.e. beta = beta_0 or 0 periodically
    
    Hyperbolic : If True, will implement tanh behaviour for Beta (converges towards beta_0)
    beta = beta_0 * 0.5 (1+ tanh( 0.001x - gamma));
    i.e. : beta = 0.5 * beta_0 at gamma * 1e3 and beta = beta_0 at ~2.1*gamma*1e3
    
    gamma gives either the period (1/frequency) for the sine or the shift for the Exp behaviour
    
    """
    def __init__(self, how='MSE', alpha=1, beta=1e-2, cyclic=False, hyperbolic = False, gamma = None):
        super(VAELoss, self).__init__()
        self.criterion = dict_[how]
        self.reshape = False
        # When using CEL, should transpose (1,2) on the x_hat and do torch.argmax(x, dim=2) 
        # But should also get rid of the positional encoding or separate it somehow
        if how == 'CEL':
            self.CEL = True
            
        self.alpha = alpha
        self.beta = beta
        self.cyclic = False
        self.hyperbolic = False
        self.gamma = 1
        self.istrain = True
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
        
    def forward(self, x_hat, x, mu, logvar):
        if self.cyclic == True : 
            self.beta = self.beta_0 * math.pow(math.sin((-0.3+ math.pi*(self.count/(self.gamma)))), 2)
            #Only steps during training
            if self.istrain == True : 
                self.count += 1
        
        if self.hyperbolic == True :
            self.beta = self.beta_0 * 0.5* (1+ math.tanh(3e-1*self.count - self.gamma))
            #Only steps during training
            if self.istrain == True :
                self.count += 1
            
        REC = self.criterion(x_hat.view(-1, x.shape[1], x.shape[2]), x)
        KLD = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        return self.alpha*REC + self.beta*KLD
    
    def train(self):
        self.istrain = True
    def eval(self):
        self.istrain = False
    

class VAELoss_vj(nn.Module):
    """
    Module defining a VAE loss combining either of the following 3 losses and a Kullback Leibler Divergence, weighted with alpha and beta respectively
    returned as : alpha * Reconstruction loss + beta * KLD
    how = 'HEL', 'MSE', 'BCE' --> HingeEmbeddingLoss, Mean Squared Error, Binary Cross Entropy
    Inherits from torch.nn.Module (so __call__ will call self.forward)
    
    Cyclic : If True, will implement sinusoid behaviour for Beta with a slight (0.3) shift:
    beta = beta_0 * sin(-0.3 + (x/gamma) * pi)^2 ; i.e. beta = beta_0 or 0 periodically
    
    Hyperbolic : If True, will implement tanh behaviour for Beta (converges towards beta_0)
    beta = beta_0 * 0.5 (1+ tanh( 0.001x - gamma));
    i.e. : beta = 0.5 * beta_0 at gamma * 1e3 and beta = beta_0 at ~2.1*gamma*1e3
    
    gamma gives either the period (1/frequency) for the sine or the shift for the Exp behaviour
    
    """
    def __init__(self, how='MSE', alpha=1, beta=1e-2, v_weight = 0.75,
                 j_weight = 0.1, cyclic=False, hyperbolic = False, gamma = None):
        super(VAELoss_vj, self).__init__()
        self.criterion = dict_[how]
        self.vj_criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.v_weight = v_weight
        self.j_weight = j_weight
        self.cyclic = False
        self.hyperbolic = False
        self.gamma = 1
        self.istrain =False
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
    
    def train(self):
        self.istrain = True
    def eval(self):
        self.istrain = False
        
    def forward(self, x_hat, x, mu, logvar):
        """x_hat and x are tuples of format (amino_acid_encoding, v_encoding, j_encoding)"""
        if self.cyclic == True : 
            self.beta = self.beta_0 * math.pow(math.sin((-0.3+ math.pi*(self.count/(self.gamma)))), 2)
            if self.istrain == True :
                self.count += 1
        
        if self.hyperbolic == True :
            self.beta = self.beta_0 * 0.5* (1+ math.tanh(3e-1*self.count - self.gamma))
            if self.istrain == True :
                self.count += 1

        #ReconstructionLoss. 
        x_hat = list(x_hat)
        x = list(x)
        REC = self.alpha * self.criterion(x_hat[0].view(-1, x[0].shape[1], x[0].shape[2]), x[0])
        #V/J loss
        VJ_LOSS = self.v_weight * self.vj_criterion(x_hat[1], x[1]) + self.j_weight * self.vj_criterion(x_hat[2], x[2])
        KLD = self.beta * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        return REC + VJ_LOSS + KLD
    
def compute_metrics(y_true, y_pred, seq_original, seq_reconstructed):
    results = {}
    #print(y_true.shape, y_pred.shape)

    results['accuracy'] = np.mean([accuracy_score(x,y) for x,y in zip(y_true,y_pred)])
    results['precision'] = precision_score(y_true, y_pred, average = 'samples')
    results['recall'] = recall_score(y_true, y_pred, average = 'samples')
    results['f1'] = f1_score(y_true, y_pred, average = 'samples')
    #results['hamming'] = hamming_loss(y_true, y_pred)
    hamm_total, hamm_seq = HammingBatch(seq_original, seq_reconstructed)
    results['hamming_total'] = hamm_total
    results['hamming_sequence'] = hamm_seq
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
