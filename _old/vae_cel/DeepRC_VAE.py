import numpy as np
import torch
import torch.nn as nn
import torch.jit as jit
from typing import List
from vae_cel.vae_cel import VAE_cel

class AttentionNetwork(nn.Module):
    def __init__(self, n_input_features: int=100, n_layers: int = 2, n_units: int = 32, dropout=False):
        """Attention network (`f()` in paper) as fully connected network.
         Currently only implemented for 1 attention head and query.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_layers : int
            Number of attention layers to compute keys
        n_units : int
            Number of units in each attention layer
        """
        super(AttentionNetwork, self).__init__()
        self.n_attention_layers = n_layers
        self.n_units = n_units
            
        fc_attention = []
        for _ in range(self.n_attention_layers):
            #Here add layers with hidden units 
            att_linear = nn.Linear(n_input_features, self.n_units)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            fc_attention.append(nn.SELU())
            #HERE ADDED NEW
            n_input_features = self.n_units
            
        #For a given sequence, take the input features and computes a score dim (40->1)
        att_linear = nn.Linear(n_input_features, 1) 
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)
    
    def forward(self, inputs):
        """Apply single-head attention network.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_input_features)
        
        Returns
        ---------
        attention_weights: torch.Tensor
            Attention weights for sequences as tensor of shape (n_sequences, 1)
        """
        attention_weights = self.attention_nn(inputs)
        return attention_weights

class OutputNetwork(nn.Module):
    def __init__(self, n_input_features: int= 100, n_output_features: int = 1, n_layers: int = 1, n_units: int = 32, dropout = False):
        """Output network (`o()` in paper) as fully connected network
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_output_features : int
            Number of output features
        n_layers : int
            Number of layers in output network (in addition to final output layer)
        n_units : int
            Number of units in each attention layer
        """
        super(OutputNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        output_network = [nn.Linear(n_input_features, 512),
                          nn.BatchNorm1d(512),
                          nn.SELU()]
        n_features = 512
        for _ in range(self.n_layers-1):
            o_linear = nn.Linear(n_features, self.n_units)
            output_network.append(o_linear)
            output_network.append(nn.SELU())
            n_features = self.n_units
        
        o_linear = nn.Linear(n_features, n_output_features)
        output_network.append(o_linear)
        self.output_nn = torch.nn.Sequential(*output_network)
        
    def forward(self, inputs):
        """Apply output network to `inputs`.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_samples, n_input_features).
        
        Returns
        ---------
        prediction: torch.Tensor
            Prediction as tensor of shape (n_samples, n_output_features).
        """
        predictions = self.output_nn(inputs)
        return predictions

class DeepRC_VAE(nn.Module): 
    
    def __init__(self, #20 for the initial onehot encode dim
                 embedding_net: torch.nn.Module = VAE_cel(seq_len = 23, aa_dim =25,
                                                          latent_dim = 100, act = nn.SELU(),
                                                          v_dim = 30, j_dim = 2),
                 attention_net: torch.nn.Module = AttentionNetwork(n_input_features = 100, # == latent_dim
                                                                   n_layers=2, n_units = 50),
                 output_net: torch.nn.Module = OutputNetwork(n_input_features=100, 
                                                             n_output_features = 2, #2 because binary class OvR
                                                              n_layers = 2, n_units = 50)
                ):
        
        super(DeepRC_VAE, self).__init__()
        #Loading VAE and freezing the weights
        self.embedding = embedding_net.to(dtype=torch.float32)
        for p in self.embedding.parameters():
            p.requires_grad = False
            
        self.attention = attention_net.to(dtype=torch.float32)
        self.output = output_net.to(dtype=torch.float32)
        
    def forward(self, x_tuple, n_per_bag):
        seq_embed = self.embedding.embed(x_tuple) # HERE x is the tuple, outputs the (V)alues vector 
        seq_attention = self.attention(seq_embed) # HERE = QK.T/scaled vector
        # BUT softmax(attn_weight) must be done PER BAG. Given we have the number of sequences,
        #we treat the input (A sequence of bags) sequentially : 
        
        emb_after_attention_per_bag = []
        start_i = 0
        # n_per_bag stores the number of sequences per bag, so we can use it to slice the attention and embedding
        for n_seqs in n_per_bag : 
            #SLICE AND SOFTMAX OVER THE SLICE
            attention_slice = torch.softmax(seq_attention[start_i:start_i+n_seqs], dim=0) # here, for a given bag/slice, compute the the Softmax(QKT)
            embedding_slice = seq_embed[start_i:start_i+n_seqs] # here get the corresponding slice
            
            embedding_attention = embedding_slice * attention_slice # final Attention computation (Softmax(QK.T/dk)*V)
            #Weighted sum over the features. The Weight is from the attention
            emb_after_attention_per_bag.append(embedding_attention.sum(dim=0))
            start_i += n_seqs
            
        x = torch.stack(emb_after_attention_per_bag, dim = 0)
        prediction_score = self.output(x)
        return prediction_score
    
