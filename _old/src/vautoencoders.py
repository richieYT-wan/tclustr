import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.preprocessing import decode_batch
# SIMPLE AUTO ENCODER which works the best apparently
class AutoEncoder(nn.Module):
    def __init__(self, seq_len, aa_dim, latent_dim, act = nn.SELU()):
        super(AutoEncoder, self).__init__()
        self.seq_len = seq_len
        self.aa_dim = aa_dim
        self.in_dim = seq_len * aa_dim
        self.lat_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), 100),
                                     act,
                                     nn.Linear(100, latent_dim)
                                    )
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim,  100),
                                     act,
                                     nn.Linear(100,  math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), self.in_dim),
                                     nn.ReLU()
                                    )
    def forward(self, x):
        encoding = self.encoder(x.view(-1, self.seq_len * self.aa_dim))
        decoding = self.decoder(encoding)
        return encoding, decoding
    
    def reconstruct_seq(self, x):
        with torch.no_grad():
            _, decoded = self(x)
            decoded = decoded.view(-1, x.shape[1], x.shape[2])
            decoded = torch.argmax(decoded, dim = 2)
            decoded = F.one_hot(decoded)
            return decode_batch(decoded)

    def embed(self,x):
        embedded = self.encoder(x.view(-1, self.seq_len*self.aa_dim))
        return embedded

# --------------------------------------------------------------------------------
class VAE_tune(nn.Module):
    def __init__(self, seq_len = 23, aa_dim = 21, latent_dim=64, act = nn.SELU()):
        super(VAE_tune, self).__init__()
        self.seq_len = seq_len
        self.aa_dim = aa_dim
        self.in_dim = seq_len * aa_dim
        self.lat_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), 100),
                                     act,
                                    )
        
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_var = nn.Linear(100, latent_dim)
        self.sig = nn.Sigmoid()
        self.decoder = nn.Sequential(nn.Linear(self.lat_dim,  100),
                                     act,
                                     nn.Linear(100,  math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), self.in_dim),
                                     )
        
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.empty_like(mu).normal_(mean=0,std=.5) 
        z = (epsilon*std) + mu
        return z
    
    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1, self.in_dim))
        mu = self.fc_mu(mu_logvar)
        logvar = self.fc_var(mu_logvar)
        return mu, logvar
    
    def embed(self, x) :
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z) # = Decoded view
        x_hat = F.relu(x_hat)
        #x = self.sig(x)
        return x_hat, mu, logvar
    
    def sample(self, n_samples):
        z = torch.randn((n_samples, self.lat_dim)).to(device=self.encoder[0].weight.device)
        return self.decode(z)
    
    
    
##################################################################

class VAE_VJ_tune(nn.Module):
    """
    Input : Flattened&concatenated AA_onehot, v_onehot, j_onehot 
    shape (N, aa_dim*max_len + v_dim + j_dim), should be (N, 25*23+30+2) when using Positional encoding
    """
    def __init__(self, seq_len = 23, aa_dim = 21, positional = True, 
                 latent_dim= 50, act = nn.SELU(), 
                 v_dim = 30, j_dim = 2, pos_dim = 4):
        
        super(VAE_VJ_tune, self).__init__()
        self.seq_len = seq_len
        self.aa_dim = aa_dim
        self.in_dim = seq_len * aa_dim + v_dim + j_dim #23*25 + 32 = 607 with positional
        self.lat_dim = latent_dim
        self.v_dim = v_dim
        self.j_dim = j_dim
        self.pos_dim = None
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, math.floor(self.in_dim/2)),
                                     act,
                                     nn.Linear(math.floor(self.in_dim/2), 100),
                                     act,
                                    )
        
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_var = nn.Linear(100, latent_dim)
        self.sig = nn.Sigmoid()
        self.decoder = nn.Sequential(nn.Linear(self.lat_dim,  100),
                                     act,
                                     nn.Linear(100,  math.floor(self.in_dim/2)),
                                     act
                                     )
        
        self.out_aa = nn.Linear(math.floor(self.in_dim/2), self.seq_len * self.aa_dim)
        #if positional == True:
        #    self.pos_dim = 4
        #    self.out_pos = nn.Linear(math.floor(self.in_dim/2), self.max_len*self.pos_dim)
            
        self.out_v = nn.Linear(math.floor(self.in_dim/2), self.v_dim)
        self.out_j = nn.Linear(math.floor(self.in_dim/2), self.j_dim)
        
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.empty_like(mu).normal_(mean=0,std=1) 
        z = (epsilon*std) + mu
        return z
    
    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1, self.in_dim))
        mu = self.fc_mu(mu_logvar)
        logvar = self.fc_var(mu_logvar)
        return mu, logvar
    
    def embed(self, x) :
        mu, logvar = self.encode(x)
        return self.reparameterise(mu, logvar)
    
    def decode(self, z):
        x_hat = self.decoder(z)
        #I can do this, or maybe just do the ReLU thing for the aa/pos reconstruction (like normal VAE)
        #then another layers for V and J
        #aa_reconstructed = self.out_aa(x)
        #aa_reconstructed = F.softmax(aa_reconstructed.view(-1, self.max_len, self.aa_dim), dim = 2)
        v_reconstructed = self.out_v(x_hat) #Logits with no activation to use with CrossEntropyLoss
        j_reconstructed = self.out_j(x_hat) #Logits with no activation to use with CrossEntropyLoss
        # FOR NOW, DO GOOD OLD RELU ON OUTPUT
        aa_reconstructed = self.out_aa(x_hat)
        
        #if self.pos_dim is not None:
        #    positional_reconstructed = self.out_pos(x)
        #    positional_reconstructed = F.softmax(positional_reconstructed.view(self.max_len,
        #                                                                       self.pos_dim), 
        #                                         dim = 2)
        #    return aa_reconstructed, positional_reconstructed, v_reconstructed, j_reconstructed
        
        return aa_reconstructed, v_reconstructed, j_reconstructed
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        xs_hat = self.decode(z) #this is a tuple, so its xs_hat instead of x
        return xs_hat, mu, logvar
    
    def sample(self, n_samples):
        z = torch.randn((n_samples, self.lat_dim)).to(device=self.encoder[0].weight.device)
        return self.decode(z)

