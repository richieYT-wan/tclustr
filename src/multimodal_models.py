import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from src.data_processing import get_positional_encoding, encoding_matrix_dict
from src.models import NetParent
import math


class Encoder(NetParent):
    """
    Encoder class ;
    Should probly not have reparameterise here because that should always be done after PoE in this set-up
    """

    def __init__(self, max_len, aa_dim, pos_dim, encoding, pad_scale, activation, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.matrix_dim = self.aa_dim + self.pos_dim
        self.encoding = encoding
        self.pad_scale = pad_scale
        input_dim = (max_len * self.matrix_dim)
        self.input_dim = input_dim
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale

        self.fc_encode = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                       nn.Linear(input_dim // 2, hidden_dim), activation)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.fc_encode(x.flatten(start_dim=1))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def slice_x(self, x):
        """
        Slices and extracts // reshapes the sequence vector
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        sequence = x[:, 0:(self.max_len * (self.aa_dim + self.pos_dim))].view(-1, self.max_len,
                                                                              (self.aa_dim + self.pos_dim))
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        # During evaluation, the trick is disabled and z = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        else:
            return mu


class Decoder(NetParent):
    def __init__(self, max_len, aa_dim, pos_dim, encoding, pad_scale, activation, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.matrix_dim = self.aa_dim + self.pos_dim
        self.encoding = encoding
        self.pad_scale = pad_scale
        input_dim = (max_len * self.matrix_dim)
        self.input_dim = input_dim
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale

        self.fc_decode = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                       nn.Linear(hidden_dim, input_dim // 2), activation,
                                       nn.Linear(input_dim // 2, input_dim))

        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))
        # Neural network params
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc_decode = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                       nn.Linear(hidden_dim, input_dim // 2), activation,
                                       nn.Linear(input_dim // 2, input_dim))

    def forward(self, z):
        x_hat = self.fc_decode(z)
        return x_hat

    def slice_x(self, x):
        """
        Slices and extracts // reshapes the sequence vector
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        sequence = x[:, 0:(self.max_len * (self.aa_dim + self.pos_dim))].view(-1, self.max_len,
                                                                              (self.aa_dim + self.pos_dim))
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.pos_dim > 0:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.fc_decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.slice_x(x_hat)
            return sequence, positional_encoding

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
        sequence, positional_encoding = self.slice_x(x_hat)
        seq_idx = self.recover_indices(sequence)
        return seq_idx, positional_encoding


class BSSVAE(NetParent):
    """
    BSS stands for BimodalSemiSupervised ; Uses a quad encoder - dual decoder architecture
    QEDC
    """

    def __init__(self, max_len_a1=0, max_len_a2=0, max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23,
                 max_len_pep=12, encoding='BL50LO', pad_scale=-20, aa_dim=20, add_positional_encoding=False,
                 activation=nn.SELU(), latent_dim=64, hidden_dim_tcr=128, hidden_dim_pep=64,
                 reparameterise_order='before'):
        super(BSSVAE, self).__init__()

        max_len_tcr = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3])
        pos_dim_tcr = sum([int(mlx) > 0 for mlx in
                           [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]]) \
            if add_positional_encoding else 0
        self.latent_dim = latent_dim
        self.activation = activation
        self.hidden_dim_tcr = hidden_dim_tcr
        self.hidden_dim_pep = hidden_dim_pep
        self.add_positional_encoding = add_positional_encoding
        self.max_len_tcr = max_len_tcr
        self.max_len_pep = max_len_pep
        self.encoding = encoding
        self.pad_scale = pad_scale
        self.aa_dim = aa_dim
        self.matrix_dim_tcr = aa_dim + pos_dim_tcr
        self.matrix_dim_pep = aa_dim

        # The "side" encoders (marginal only, Ztcr and Zpep)
        self.marg_tcr_encoder = Encoder(max_len_tcr, aa_dim, pos_dim_tcr, encoding, pad_scale, activation,
                                        hidden_dim_tcr, latent_dim)
        # pos_dim_pep is 0, "useless" to encode positional encoding for a single chain
        self.marg_pep_encoder = Encoder(max_len_pep, aa_dim, 0, encoding, pad_scale, activation,
                                        hidden_dim_pep, latent_dim)

        # The "middle" encoders (joint, Ztcr,pep) through PoE or else ;
        self.joint_tcr_encoder = Encoder(max_len_tcr, aa_dim, pos_dim_tcr, encoding, pad_scale, activation,
                                         hidden_dim_tcr, latent_dim)
        self.joint_pep_encoder = Encoder(max_len_pep, aa_dim, 0, encoding, pad_scale, activation,
                                         hidden_dim_pep, latent_dim)

        # Dual decoders that are shared among modalities
        self.tcr_decoder = Decoder(max_len_tcr, aa_dim, pos_dim_tcr, encoding, pad_scale, activation,
                                   hidden_dim_tcr, latent_dim)
        self.pep_decoder = Decoder(max_len_pep, aa_dim, 0, encoding, pad_scale, activation,
                                   hidden_dim_pep, latent_dim)

    def forward(self, x_tcr_marg, x_tcr_joint, x_pep_joint, x_pep_marg):
        """
        # TODO: At some point: refactor, probably return a dictionary instead because this is bound to be wrong with list orders...
        Args:
            x_tcr_marg:
            x_tcr_joint:
            x_pep_joint:
            x_pep_marg:

        Returns:
            recons (list[torch.Tensor]): [recon_tcr_marg, recon_tcr_joint, recon_pep_marg, recon_pep_joint]
            mus (list[torch.Tensor]): [mu_tcr_poe, mu_pep_poe, mu_joint_poe]
            logvars (list[torch.Tensor]): [logvar_tcr_poe, logvar_pep_poe, logvar_joint_poe]
        """
        #####################################################
        # BOTTOM HALF OF THE GRAPH ; ENCODING PART WITH POE #
        #####################################################

        # "order" : marg tcr -> marg pep -> joint (tcr -> pep)
        # Running each encoder-decoder pair to get reconstructed, mu, logvars
        print(self.device, x_tcr_marg.device)
        mu_tcr_marg, logvar_tcr_marg = self.marg_tcr_encoder(x_tcr_marg)
        mu_pep_marg, logvar_pep_marg = self.marg_pep_encoder(x_pep_marg)
        # The "joint" TCR / Pep inputs are those coming from the paired input
        mu_tcr_joint, logvar_tcr_joint = self.joint_tcr_encoder(x_tcr_joint)
        mu_pep_joint, logvar_pep_joint = self.joint_pep_encoder(x_pep_joint)
        # latent (Zs) with Product of Experts MARGINAL
        # Z_tcr
        mu_tcr_poe_marg, logvar_tcr_poe_marg = self.product_of_experts([mu_tcr_marg, mu_tcr_joint],
                                                                       [logvar_tcr_marg, logvar_tcr_joint])
        # Z_pep
        mu_pep_poe_marg, logvar_pep_poe_marg = self.product_of_experts([mu_pep_marg, mu_pep_joint],
                                                                       [logvar_pep_marg, logvar_pep_joint])
        # LATENT with PoE JOINT (tcr_joint, pep_joint)
        # Z_tcr,pep
        mu_joint_poe, logvar_joint_poe = self.product_of_experts([mu_tcr_joint, mu_pep_joint],
                                                                 [logvar_tcr_joint, logvar_pep_joint])

        #####################################################
        # UPPER HALF OF THE GRAPH  ; DECODING PART WITH POE #
        #####################################################

        # Get 3 latents (2 marginals and 1 joint)
        z_tcr_marg = self.reparameterise(mu_tcr_poe_marg, logvar_tcr_poe_marg)
        z_joint = self.reparameterise(mu_joint_poe, logvar_joint_poe)
        z_pep_marg = self.reparameterise(mu_pep_poe_marg, logvar_pep_poe_marg)
        # Get 4 reconstruction (1 from each Marginal->PoE, 2 from joint(tcr+pep))
        recon_tcr_marg = self.tcr_decoder(z_tcr_marg)
        recon_tcr_joint = self.tcr_decoder(z_joint)
        recon_pep_joint = self.pep_decoder(z_joint)
        recon_pep_marg = self.pep_decoder(z_pep_marg)

        # Return all outputs using dictionaries for splitting modalities it's tidier than lists
        recons = {'tcr_marg': recon_tcr_marg,
                  'tcr_joint': recon_tcr_joint,
                  'pep_joint': recon_pep_joint,
                  'pep_marg': recon_pep_marg}
        mus = {'tcr_marg': mu_tcr_poe_marg,
               'joint': mu_joint_poe,
               'pep_marg': mu_pep_poe_marg}
        logvars = {'tcr_marg': logvar_tcr_poe_marg,
                   'joint': logvar_joint_poe,
                   'pep_marg': logvar_pep_poe_marg}
        # recons = [recon_tcr_marg, recon_tcr_joint, recon_pep_joint, recon_pep_marg]
        # mus = [mu_tcr_poe_marg, mu_joint_poe, mu_pep_poe_marg]
        # logvars = [logvar_tcr_poe_marg, logvar_joint_poe, logvar_pep_poe_marg]
        return recons, mus, logvars

    @staticmethod
    def product_of_experts(mus_list, logvars_list):
        var = torch.exp(torch.stack(logvars_list))
        mu = torch.stack(mus_list)
        # Precision = 1/ vars
        T = 1 / var
        mu_poe = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        var_poe = 1 / torch.sum(T, dim=0)
        logvar_poe = torch.log(var_poe)
        return mu_poe, logvar_poe

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        # During evaluation, the trick is disabled and z = mu
        else:
            return mu

    def embed(self, x_a, x_b, which):
        """
        Not the best way but assumes the following:
        # TODO: This is wrong for the marginal cases!! FIX IT
        if marg :
            x_a = marg, x_b = joint
        if joint:
            x_a = tcr, x_b = pep
        Args:
            x_a:
            x_b:
            which:

        Returns:
        """
        if which == 'tcr':
            mu_a, logvar_a = self.marg_tcr_encoder(x_a)
            mu_b, logvar_b = self.joint_tcr_encoder(x_b)
        elif which == 'pep':
            mu_a, logvar_a = self.marg_pep_encoder(x_a)
            mu_b, logvar_b = self.joint_pep_encoder(x_b)
        elif which == 'joint':
            mu_a, logvar_a = self.joint_tcr_encoder(x_a)
            mu_b, logvar_b = self.joint_pep_encoder(x_b)

        mu, logvar = self.product_of_experts([mu_a, mu_b], [logvar_a, logvar_b])
        z = self.reparameterise(mu, logvar)
        return z

    def forward_marginal(self, x, which):
        if which == 'tcr':
            mu_marg, logvar_marg = self.marg_tcr_encoder(x)
            mu_joint, logvar_joint = self.joint_tcr_encoder(x)
        elif which == 'pep':
            mu_marg, logvar_marg = self.marg_pep_encoder(x)
            mu_joint, logvar_joint = self.joint_pep_encoder(x)
        mu, logvar = self.product_of_experts([mu_marg, mu_joint], [logvar_marg, logvar_joint])
        z = self.reparameterise(mu, logvar)
        x_hat = self.tcr_decoder(z) if which == 'tcr' else self.pep_decoder(z)
        return x_hat, z

    def forward_joint(self, x_tcr, x_pep):
        mu_tcr, logvar_tcr = self.joint_tcr_encoder(x_tcr)
        mu_pep, logvar_pep = self.joint_pep_encoder(x_pep)
        mu, logvar = self.product_of_experts([mu_tcr, mu_pep], [logvar_tcr, logvar_pep])
        z = self.reparameterise(mu, logvar)
        x_hat_tcr = self.tcr_decoder(z)
        x_hat_pep = self.pep_decoder(z)
        return x_hat_tcr, x_hat_pep, z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z

    # Reconstruction QOL methods
    def slice_x(self, x, which):
        if which == 'tcr':
            return self.tcr_decoder.slice_x(x)
        elif which == 'pep':
            return self.pep_decoder.slice_x(x)

    def reconstruct(self, z, which):
        if which == 'tcr':
            return self.tcr_decoder.reconstruct(z)
        elif which == 'pep':
            return self.pep_decoder.reconstruct(z)

    def recover_indices(self, seq_tensor):
        # Simply call one of the children to do this ; this doesn't need/affect weights
        # and is simply a tensor2seq reconstruction
        return self.tcr_decoder.recover_indices(seq_tensor)

    def recover_sequences_blosum(self, seq_tensor, AA_KEYS='ARNDCQEGHILKMFPSTWYVX'):
        """
        Args:
            seq_tensor:
            AA_KEYS:

        Returns: list_of_sequences (list of str)

        """
        return self.tcr_decoder.recover_sequences_blosum(seq_tensor, AA_KEYS)

    def reconstruct_hat(self, x_hat, which):
        """
        Args:
            x_hat:
            which:

        Returns:
            seq_idx, positional_encoding
        """
        if which == 'tcr':
            return self.tcr_decoder.reconstruct_hat(x_hat)
        elif which == 'pep':
            return self.pep_decoder.reconstruct_hat(x_hat)

    def to(self, device):
        self.device = device
        for c in self.children():
            if hasattr(c, 'device') and hasattr(c, 'to'):
                c.to(device)


# Old multimodal
class SequenceVAE(NetParent):
    """
        Class to be used in multimodal VAE ; This constructs a VAE for a given sequence
        Ex: Alpha (A1 to A3 concatenated) or Peptide
        With slightly fewer parameters in the decoder with one fewer layer
    """

    # Define the input dimension as some combination of sequence length, AA dim,
    def __init__(self, max_len, encoding='BL50LO', pad_scale=-20, aa_dim=20, add_positional_encoding=False,
                 activation=nn.SELU(), hidden_dim=128, latent_dim=64):
        super(SequenceVAE, self).__init__()
        # Init params that will be needed at some point for reconstruction
        # Here, define aa_dim and pos_dim ; pos_dim is the dimension of a positional encoding.
        # pos_dim should be given by how many sequences are used, i.e. how many max_len_x > 0
        # But also use a flag `add_positional_encoding` to make it more explicit that it's active or not
        # NOTE : pos_dim does not really make sense in this context ;
        pos_dim = 1 if add_positional_encoding else 0
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.add_positional_encoding = add_positional_encoding
        input_dim = max_len * (aa_dim + pos_dim)
        self.input_dim = input_dim
        self.max_len = max_len
        self.encoding = encoding
        if pad_scale is None:
            self.pad_scale = -20 if encoding in ['BL50LO', 'BL62LO'] else 0
        else:
            self.pad_scale = pad_scale
        MATRIX_VALUES = deepcopy(encoding_matrix_dict[encoding])
        MATRIX_VALUES['X'] = np.array([self.pad_scale]).repeat(20)
        self.MATRIX_VALUES = torch.from_numpy(np.stack(list(MATRIX_VALUES.values()), axis=0))
        # Neural network params
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder : in -> in//2 -> hidden -> latent_mu, latent_logvar, where z = mu + logvar*epsilon
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim // 2), activation,
                                     nn.Linear(input_dim // 2, hidden_dim), activation)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                     nn.Linear(hidden_dim, input_dim // 2), activation,
                                     nn.Linear(input_dim // 2, input_dim))

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        # During evaluation, the trick is disabled and z = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        else:
            return mu

    def encode(self, x):
        mu_logvar = self.encoder(x.flatten(start_dim=1))
        mu = self.encoder_mu(mu_logvar)
        logvar = self.encoder_logvar(mu_logvar)
        return mu, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def slice_x(self, x):
        """
        Slices and extracts // reshapes the sequence vector
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        sequence = x[:, 0:(self.max_len * (self.aa_dim + self.pos_dim))].view(-1, self.max_len,
                                                                              (self.aa_dim + self.pos_dim))
        positional_encoding = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def reconstruct(self, z):
        with torch.no_grad():
            x_hat = self.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.slice_x(x_hat)
            return sequence, positional_encoding

    def embed(self, x):
        # by design, embed would return "mu" if self.training is False
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
        sequence, positional_encoding = self.slice_x(x_hat)
        seq_idx = self.recover_indices(sequence)
        return seq_idx, positional_encoding


class TrimodalPepTCRVAE(NetParent):
    """
        In this current implementation, we consider the cases where the input can be paired and unpaired.
        Peptide is always present, then the sequence input can either be : Alpha, Beta, Alpha+Beta, but
        the paired case is still handled with the same two networks as the uni-modal cases, i.e. there is
        not a separate VAE that handles a concatenated Alpha+Beta chain.
        [See figure 1] To replicate SVAE from Svetlana's paper, would need to have 4 encoders
        (alpha_joint, alpha_marg, beta_joint, beta_marg),
        and 2 decoders
        (alpha, beta)
    """

    # Sum of the max len of each chains
    def __init__(self, alpha_dim=7 + 8 + 22, beta_dim=6 + 7 + 23, pep_dim=12,
                 encoding='BL50LO', pad_scale=-20, aa_dim=20, add_positional_encoding=False,
                 activation=nn.SELU(),
                 hidden_dim_alpha=64, hidden_dim_beta=64, hidden_dim_pep=32,
                 latent_dim=32):
        super(TrimodalPepTCRVAE, self).__init__()
        self.vae_alpha = SequenceVAE(alpha_dim, encoding, pad_scale, aa_dim, add_positional_encoding,
                                     activation, hidden_dim_alpha, latent_dim)
        self.vae_beta = SequenceVAE(beta_dim, encoding, pad_scale, aa_dim, add_positional_encoding,
                                    activation, hidden_dim_beta, latent_dim)
        self.vae_pep = SequenceVAE(pep_dim, encoding, pad_scale, aa_dim, add_positional_encoding,
                                   activation, hidden_dim_pep, latent_dim)

    def sanity_check(self):
        if any([self.vae_beta.encoder[0].weight.isnan().any(),
                self.vae_alpha.encoder[0].weight.isnan().any(),
                self.vae_pep.encoder[0].weight.isnan().any()]):
            print(self.vae_beta.encoder[0].weight)
            print(self.vae_alpha.encoder[0].weight)
            print(self.vae_pep.encoder[0].weight)

    # Here, either separate tensors and handle the split in train_eval
    # or a single tensor and handle the split here ; Or maybe the mask should be used in metrics / loss
    def forward(self, x_alpha, x_beta, x_pep):
        # Get each latent
        self.sanity_check()
        mu_alpha, logvar_alpha = self.vae_alpha.encode(x_alpha)
        mu_beta, logvar_beta = self.vae_beta.encode(x_beta)
        mu_pep, logvar_pep = self.vae_pep.encode(x_pep)
        # Return the marginal distributions
        mus = [mu_alpha, mu_beta, mu_pep]
        logvars = [logvar_alpha, logvar_beta, logvar_pep]
        # Get the joint distribution from a product of experts to join the modalities
        mu_joint, logvar_joint = self.product_of_experts(mus,
                                                         logvars)

        # Decode from the joint latent to capture information shared by the modalities
        z = self.reparameterise(mu_joint, logvar_joint)
        # The decoding part should also handle the missing modalities somehow, i.e. disable gradients?
        # Otherwise, it will be very easy overfit by having the decoder return only "XXXXXXX" ?
        # Also need to adapt the XXXXX thing for the accuracy computation ?
        recon_alpha = self.vae_alpha.decode(z)
        recon_beta = self.vae_beta.decode(z)
        recon_pep = self.vae_pep.decode(z)
        if any([x.isnan().any() for x in [mu_alpha, mu_beta, mu_pep]]):
            print('_alpha', x_alpha.isnan().any(), mu_alpha.isnan().any(), mu_alpha)
            print('_beta', x_beta.isnan().any(), mu_beta.isnan().any(), mu_beta)
            print('_pep', x_pep.isnan().any(), mu_pep.isnan().any(), mu_pep)
            import sys
            sys.exit(1)
        # return the reconstructed, joint distribution and marginal distributions with masked modalities
        return recon_alpha, recon_beta, recon_pep, mu_joint, logvar_joint, mus, logvars

    def reconstruct_sequence(self, x_hat, which):
        m = {'alpha': self.vae_alpha, 'beta': self.vae_beta, 'pep': self.vae_pep}[which]
        sequence, positional_encoding = m.slice_x(x_hat)
        return m.recover_sequences_blosum(sequence)

    # TODO : change this to either have 3 reconstruct or use "which" to reconstruct either 3 seq
    def reconstruct_alpha(self, z):
        with torch.no_grad():
            x_hat = self.vae_alpha.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.vae_alpha.slice_x(x_hat)
            return sequence, positional_encoding

    def reconstruct_beta(self, z):
        with torch.no_grad():
            x_hat = self.vae_beta.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.vae_beta.slice_x(x_hat)
            return sequence, positional_encoding

    def reconstruct_pep(self, z):
        with torch.no_grad():
            x_hat = self.vae_pep.decode(z)
            # Reconstruct and unflattens the sequence
            sequence, positional_encoding = self.vae_pep.slice_x(x_hat)
            return sequence, positional_encoding

    @staticmethod
    def product_of_experts(mus_list, logvars_list):
        vars = torch.exp(torch.stack(logvars_list))
        mus = torch.stack(mus_list)
        # Precision = 1/ vars
        T = 1 / vars
        mu_poe = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
        var_poe = 1 / torch.sum(T, dim=0)
        logvar_poe = torch.log(var_poe)
        return mu_poe, logvar_poe

    def reparameterise(self, mu, logvar):
        # During training, the reparameterisation leads to z = mu + std * eps
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.empty_like(mu).normal_(mean=0, std=1)
            return (epsilon * std) + mu
        # During evaluation, the trick is disabled and z = mu
        else:
            return mu

    def embed(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def sample_latent(self, n_samples):
        z = torch.randn((n_samples, self.latent_dim)).to(device=self.encoder[0].weight.device)
        return z
