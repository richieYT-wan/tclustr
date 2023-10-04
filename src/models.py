import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from src.data_processing import get_positional_encoding, encoding_matrix_dict
import math


# import wandb

class NetParent(nn.Module):
    """
    Mostly a QOL superclass
    Creates a parent class that has reset_parameters implemented and .device
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()
        # device is cpu by default
        self.device = 'cpu'

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    @staticmethod
    def reset_weight(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            if hasattr(child, 'reset_parameters'):
                self.reset_weight(child)

    def to(self, device):
        # Work around, so we can get model.device for all NetParent
        #
        super(NetParent, self).to(device)
        self.device = device


class CDR3bVAE(NetParent):
    # Define the input dimension as some combination of sequence length, AA dim,
    def __init__(self, max_len=23, encoding='BL50LO', pad_scale=-12, aa_dim=20, use_v=True, use_j=True, v_dim=51,
                 j_dim=13, activation=nn.SELU(), hidden_dim=128, latent_dim=32, max_len_pep=0):
        super(CDR3bVAE, self).__init__()
        # Init params that will be needed at some point for reconstruction
        v_dim = v_dim if use_v else 0
        j_dim = j_dim if use_j else 0
        max_len = max_len + max_len_pep
        input_dim = (max_len * aa_dim) + v_dim + j_dim
        self.encoding = encoding
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale
        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))
        self.input_dim = input_dim
        self.max_len = max_len
        print(self.max_len)
        self.aa_dim = aa_dim
        self.v_dim = v_dim if use_v else 0
        self.use_v = use_v
        self.j_dim = j_dim if use_j else 0
        self.use_j = use_j
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: For now, just use a fixed set of layers.
        # Encoder : in -> in//2 -> hidden -> latent_mu, latent_logvar, where z = mu + logvar*epsilon
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                     nn.Linear(input_dim // 2, hidden_dim), activation)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        # TODO: Maybe split the decoder into parts for seq, v, j and also update behaviour in forward etc.
        # Decoder: latent (z) -> hidden -> in // 2 -> in
        # self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
        #                              nn.Linear(hidden_dim, input_dim //2), activation,
        #                              nn.Linear(input_dim // 2, input_dim))

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                     nn.Linear(hidden_dim, hidden_dim), activation)
        # nn.Linear(input_dim // 2, input_dim))

        self.decoder_sequence = nn.Sequential(nn.Linear(hidden_dim, input_dim // 2), activation,
                                              nn.Linear(input_dim // 2, input_dim - self.v_dim - self.j_dim))

        self.decoder_v = nn.Linear(hidden_dim, self.v_dim) if use_v else None
        self.decoder_j = nn.Linear(hidden_dim, self.j_dim) if use_j else None

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
        z = (epsilon * std) + mu
        return z

    def encode(self, x):
        mu_logvar = self.encoder(x.flatten(start_dim=1))
        mu = self.encoder_mu(mu_logvar)
        logvar = self.encoder_logvar(mu_logvar)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        x_hat = self.decoder_sequence(z)
        if self.use_v:
            v = self.decoder_v(z)
            x_hat = torch.cat([x_hat, v], dim=1)
        if self.use_j:
            j = self.decoder_j(z)
            x_hat = torch.cat([x_hat, j], dim=1)
        return x_hat

    def slice_x(self, x):
        sequence = x[:, 0:(self.max_len * self.aa_dim)].view(-1, self.max_len, self.aa_dim)
        # Reconstructs the v/j gene as one hot vectors
        v_gene = x[:, (self.max_len * self.aa_dim):(self.max_len * self.aa_dim + self.v_dim)] if self.use_v else None
        j_gene = x[:, ((self.max_len * self.aa_dim) + self.v_dim):] if self.use_j else None
        return sequence, v_gene, j_gene

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, v_gene, j_gene = self.slice_x(x_hat)
            return sequence, v_gene, j_gene

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def embed(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    def recover_indices(self, seq_tensor):
        # Sample data
        N, max_len = seq_tensor.shape[0], seq_tensor.shape[1]

        # Expand MATRIX_VALUES to have the same shape as x_seq for broadcasting
        expanded_MATRIX_VALUES = self.MATRIX_VALUES.unsqueeze(0).expand(N, -1, -1, -1)
        # Compute the absolute differences
        abs_diff = torch.abs(seq_tensor.unsqueeze(2) - expanded_MATRIX_VALUES)
        # Sum along the last dimension (20) to get the absolute differences for each character
        abs_diff_sum = abs_diff.sum(dim=-1)

        # Find the argmin along the character dimension (21)
        argmin_indices = torch.argmin(abs_diff_sum, dim=-1)
        return argmin_indices

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        return [''.join([AA_KEYS[y] for y in x]) for x in self.recover_indices(seq_tensor)]

    def reconstruct_hat(self, x_hat):
        seq, v, j = self.slice_x(x_hat)
        seq_idx = self.recover_indices(seq)
        return seq_idx, v, j


class PairedFVAE(NetParent):
    # Define the input dimension as some combination of sequence length, AA dim,
    def __init__(self, max_len_b=23, max_len_a=24, max_len_pep=12, encoding='BL50LO', pad_scale=-20,
                 use_b=True, use_a=True, use_pep=False, use_v=False, use_j=False,
                 aa_dim=20, v_dim=51, j_dim=13,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=64):
        super(PairedFVAE, self).__init__()
        # Init dimensions
        v_dim = v_dim if use_v else 0
        j_dim = j_dim if use_j else 0
        b_dim = max_len_b * aa_dim if use_b else 0
        a_dim = max_len_a * aa_dim if use_a else 0
        pep_dim = max_len_pep * aa_dim if use_pep else 0
        input_dim = b_dim + a_dim + pep_dim + v_dim + j_dim
        self.seq_dim = b_dim + a_dim + pep_dim
        self.input_dim = input_dim
        self.max_len_b = max_len_b if use_b else 0
        self.use_b = use_b
        self.max_len_a = max_len_a if use_a else 0
        self.use_a = use_a
        self.max_len_pep = max_len_pep if use_pep else 0
        self.use_pep = use_pep
        self.alpha_dim = a_dim
        self.beta_dim = b_dim
        self.pep_dim = pep_dim
        self.aa_dim = aa_dim
        self.v_dim = v_dim if use_v else 0
        self.use_v = use_v
        self.j_dim = j_dim if use_j else 0
        self.use_j = use_j
        self.encoding = encoding
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale
        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # TODO: For now, just use a fixed set of layers.
        # Encoder : in -> in//2 -> hidden -> latent_mu, latent_logvar, where z = mu + logvar*epsilon
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                     nn.Linear(input_dim // 2, hidden_dim), activation)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        # TODO: Maybe split the decoder into parts for seq, v, j and also update behaviour in forward etc.
        # Decoder: latent (z) -> hidden -> in // 2 -> in
        # self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
        #                              nn.Linear(hidden_dim, input_dim //2), activation,
        #                              nn.Linear(input_dim // 2, input_dim))

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                     nn.Linear(hidden_dim, hidden_dim), activation)

        self.decoder_beta = nn.Sequential(nn.Linear(hidden_dim, input_dim // 2), activation,
                                          nn.Linear(input_dim // 2, self.beta_dim)) if use_b else None
        self.decoder_alpha = nn.Sequential(nn.Linear(hidden_dim, input_dim // 2), activation,
                                           nn.Linear(input_dim // 2, self.alpha_dim)) if use_a else None
        self.decoder_pep = nn.Sequential(nn.Linear(hidden_dim, self.pep_dim)) if use_pep else None

        self.decoder_v = nn.Linear(hidden_dim, self.v_dim) if use_v else None
        self.decoder_j = nn.Linear(hidden_dim, self.j_dim) if use_j else None

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
        z = (epsilon * std) + mu
        return z

    def encode(self, x):
        mu_logvar = self.encoder(x.flatten(start_dim=1))
        mu = self.encoder_mu(mu_logvar)
        logvar = self.encoder_logvar(mu_logvar)
        return mu, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat_b = self.decoder_beta(x_hat) if self.use_b else torch.empty([len(z), 0])
        x_hat_a = self.decoder_alpha(x_hat) if self.use_a else torch.empty([len(z), 0])
        x_hat_pep = self.decoder_pep(x_hat) if self.use_pep else torch.empty([len(z), 0])
        x_hat_v = self.decoder_v(x_hat) if self.use_v else torch.empty([len(z), 0])
        x_hat_j = self.decoder_j(x_hat) if self.use_j else torch.empty([len(z), 0])
        return torch.cat([x_hat_b, x_hat_a, x_hat_pep, x_hat_v, x_hat_j], dim=1)

    def slice_x(self, x):
        # Slices the vector values for the sequences and reconstructs (view) as 3 (Nx2D) tensors
        sequence = x[:, 0:self.seq_dim]
        seq_b = sequence[:, :self.beta_dim]\
            .view(-1, self.max_len_b, self.aa_dim) if self.use_b else None

        seq_a = sequence[:, self.beta_dim:(self.beta_dim + self.alpha_dim)] \
            .view(-1, self.max_len_a, self.aa_dim) if self.use_a else None

        seq_pep = sequence[:, (self.beta_dim + self.alpha_dim):(self.alpha_dim + self.beta_dim + self.pep_dim)] \
            .view(-1, self.max_len_pep, self.aa_dim) if self.use_pep else None

        # Reconstructs the v/j gene as one hot vectors
        v_gene = x[:, self.seq_dim:(self.seq_dim + self.v_dim)] if self.use_v else None
        j_gene = x[:, (self.seq_dim + self.v_dim):] if self.use_j else None
        # Here, returns the entire concatenated sequence ; Maybe it should be
        return (seq_b, seq_a, seq_pep), v_gene, j_gene

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.decode(z)
            # Reconstruct and unflattens the sequence
            (seq_b, seq_a, seq_pep), v_gene, j_gene = self.slice_x(x_hat)
            return seq_b, seq_a, seq_pep, v_gene, j_gene

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def embed(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    def recover_indices(self, seq_tensor):
        # Sample data
        N, max_len = seq_tensor.shape[0], seq_tensor.shape[1]

        # Expand MATRIX_VALUES to have the same shape as x_seq for broadcasting
        expanded_MATRIX_VALUES = self.MATRIX_VALUES.unsqueeze(0).expand(N, -1, -1, -1)
        # Compute the absolute differences
        abs_diff = torch.abs(seq_tensor.unsqueeze(2) - expanded_MATRIX_VALUES)
        # Sum along the last dimension (20) to get the absolute differences for each character
        abs_diff_sum = abs_diff.sum(dim=-1)

        # Find the argmin along the character dimension (21)
        argmin_indices = torch.argmin(abs_diff_sum, dim=-1)
        return argmin_indices

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        return [''.join([AA_KEYS[y] for y in x]) for x in self.recover_indices(seq_tensor)]

    def reconstruct_hat(self, x_hat):
        (seq_b, seq_a, seq_pep), v, j = self.slice_x(x_hat)
        seq_idx_b = self.recover_indices(seq_b) if self.use_b else torch.empty([len(x_hat),0])
        seq_idx_a = self.recover_indices(seq_a) if self.use_a else torch.empty([len(x_hat),0])
        seq_idx_pep = self.recover_indices(seq_pep) if self.use_pep else torch.empty([len(x_hat),0])
        seq = torch.cat([seq_idx_b, seq_idx_a, seq_idx_pep], dim=1)
        return seq, v, j


# STANDARDIZERS

class StandardizerSequence(nn.Module):
    def __init__(self, n_feats=20):
        super(StandardizerSequence, self).__init__()
        # Here using 20 because 20 AA alphabet. With this implementation, it shouldn't need custom state_dict fct
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats
        self.dimensions = None

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = (torch.sum(masked_values, dim=1) / torch.sum(x_mask, dim=1))
            sigma = (torch.sqrt(torch.sum((masked_values - mu.unsqueeze(1)) ** 2, dim=1) / torch.sum(x_mask, dim=1)))
            self.mu.data.copy_(mu.mean(dim=0))
            sigma = sigma.mean(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = (self.view_3d_to_2d(x) - self.mu) / self.sigma
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def recover(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = self.view_3d_to_2d(x)
            # Return to original scale by multiplying with sigma and adding mu
            x = x * self.sigma + self.mu
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros(self.n_feats))
            self.sigma.data.copy_(torch.ones(self.n_feats))
            self.fitted.data = torch.tensor(False)

    def view_3d_to_2d(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                self.dimensions = (x.shape[0], x.shape[1], x.shape[2])
                return x.view(-1, x.shape[2])
            else:
                return x

    def view_2d_to_3d(self, x):
        with torch.no_grad():
            if len(x.shape) == 2 and self.dimensions is not None:
                return x.view(self.dimensions[0], self.dimensions[1], self.dimensions[2])
            else:
                return x


class StandardizerSequenceVector(nn.Module):
    def __init__(self, input_dim=20, max_len=12):
        super(StandardizerSequenceVector, self).__init__()
        self.mu = nn.Parameter(torch.zeros((max_len, input_dim)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((max_len, input_dim)), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.input_dim = input_dim
        self.max_len = max_len

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = masked_values.mean(dim=0)
            sigma = masked_values.std(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.mu.data.copy_(mu)
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'Standardizer not fitted!'
        return (x - self.mu) / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros((self.max_len, self.input_dim)))
            self.sigma.data.copy_(torch.ones((self.max_len, self.input_dim)))
            self.fitted.data = torch.tensor(False)


class StandardizerFeatures(nn.Module):
    def __init__(self, n_feats=2):
        super(StandardizerFeatures, self).__init__()
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats

    def fit(self, x_features: torch.Tensor):
        """ Will consider the mask (padded position) and ignore them before computing the mean/std
        Args:
            x_features:

        Returns:
            None
        """
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            self.mu.data.copy_(x_features.mean(dim=0))
            self.sigma.data.copy_(x_features.std(dim=0))
            # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
            self.sigma.data[torch.where(self.sigma.data == 0)] = 1e-12
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            return x - self.mu / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy(torch.zeros(self.n_feats))
            self.sigma.data.copy(torch.ones(self.n_feats))
            self.fitted.data = torch.tensor(False)


class StdBypass(nn.Module):
    def __init__(self, **kwargs):
        super(StdBypass, self).__init__()
        self.requires_grad = False
        self.bypass = nn.Identity(**kwargs)
        self.fitted = False
        self.mu = 0
        self.sigma = 1

    def forward(self, x_tensor, *args):
        """
        Args:
            x:
        Returns:

        """

        return x_tensor

    def fit(self, x_tensor, *args):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """
        self.fitted = True
        return x_tensor
