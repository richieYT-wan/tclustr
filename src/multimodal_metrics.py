import math
from typing import List, Dict, Tuple

import torch
from overrides import override
from torch import nn
from torch.nn import functional as F

from src.metrics import LossParent


class MultimodalLossParent(LossParent):
    def __init__(self, max_len_tcr, max_len_pep,
                 aa_dim=20, add_positional_encoding=False, positional_weighting=True,
                 sequence_criterion=nn.MSELoss(reduction='none'), weight_seq=1,
                 weight_kld_z=1, weight_kld_n=1e-2,
                 kld_warm_up=100, kld_tanh_scale=0.1, kld_decrease=5e-3, flat_phase=None, debug=False):
        super(MultimodalLossParent, self).__init__(debug)
        self.max_len_tcr = max_len_tcr
        self.max_len_pep = max_len_pep
        self.weight_seq = 1
        self.base_weight_kld_n = weight_kld_n
        self.weight_kld_n = 0
        self.base_weight_kld_z = weight_kld_z
        self.weight_kld_z = weight_kld_z
        self.kld_warm_up = kld_warm_up
        self.kld_tanh_scale = kld_tanh_scale
        self.flat_phase = kld_warm_up // 3 if flat_phase is None else flat_phase
        self.kld_decrease = kld_decrease
        self.counter = 0

        assert sequence_criterion.reduction == 'none', f'Reduction mode should be "none" for sequence criterion! Got {sequence_criterion.reduction} instead.'

        # Defining dimensions for vectors
        self.aa_dim = aa_dim
        # Dimension for positional weighting if applicable
        self.pos_dim_tcr = 1 if add_positional_encoding else 0
        self.add_positional_encoding = add_positional_encoding
        # Reconstruction part
        self.weight_seq = weight_seq
        self.sequence_criterion = sequence_criterion
        # Positional weighting to set weights for CDR3 to 3x while CDR1,2 weights are at 1
        self.positional_weighting = positional_weighting
        self.positional_weights = torch.ones([max_len_tcr, 20]).to(self.device)

    def reconstruction_loss(self, x_hat, x, which):
        """
        Uses slicing to reshape the vectors and apply the reconstruction loss
        Args:
            x_hat:
            x:
            which: Defines which vector we are slicing-reshaping. Should be 'pep' or 'tcr'

        Returns: mean loss for current batch x, x_hat
        """
        x_hat_seq, positional_hat = self.slice_x(x_hat, which)
        x_true_seq, positional_true = self.slice_x(x, which)
        reconstruction_loss = self.sequence_criterion(x_hat_seq, x_true_seq)
        # if positional weighting, then multiply the loss to give larger/smaller gradients w.r.t. chains and positions
        if self.positional_weighting and which == 'tcr':
            reconstruction_loss = reconstruction_loss.mul(self.positional_weights)

        # Here, take the mean before checking for positional encoding because we have un-reduced loss
        # and scale it by the "weight_seq" (versus weight_kld)
        reconstruction_loss = self.weight_seq * reconstruction_loss.mean()

        # check if we use positional encoding, if yes we add the loss
        if self.add_positional_encoding:
            # Here use 5e-4, very small weight because this task should be trivial
            positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true)
            reconstruction_loss += (5e-4 * self.weight_seq * positional_loss)

        return reconstruction_loss

    def kld_latent(self, mu_1, logvar_1, mu_2, logvar_2):
        """
        Compute the Kullback-Leibler Divergence between two Gaussian distributions.

        Parameters:
        - mu1: Mean of the first distribution.
        - logvar1: Log variance of the first distribution.
        - mu2: Mean of the second distribution.
        - logvar2: Log variance of the second distribution.

        Returns:
        - The KLD from Q to P.
        """
        sigma1_squared = logvar_1.exp()
        sigma2_squared = logvar_2.exp()
        kld = 0.5 * torch.sum(
            logvar_1 - logvar_2 - 1 + sigma2_squared / sigma1_squared + (mu_2 - mu_1).pow(2) / sigma1_squared, dim=-1)
        return self.weight_kld_z * kld.mean()

    def kld_normal(self, mu, logvar):
        kld = 1 + logvar - mu.pow(2) - logvar.exp()
        return self.weight_kld_n * (-0.5 * torch.mean(kld))

    # class utils methods
    def slice_x(self, x, which):
        if which == 'tcr':
            sequence = x[:, 0:self.max_len_tcr * (self.aa_dim + self.pos_dim_tcr)].view(-1, self.max_len_tcr,
                                                                                        self.aa_dim + self.pos_dim_tcr)
        elif which == 'pep':
            sequence = x[:, 0:self.max_len_pep * self.aa_dim].view(-1, self.max_len_pep, self.aa_dim)
        positional_encoding = None
        if self.add_positional_encoding:
            positional_encoding = sequence[:, :, self.aa_dim:]
            sequence = sequence[:, :, :self.aa_dim]
        return sequence, positional_encoding

    def _kld_weight_regime(self):
        """
        if counter <= warm_up : tanh annealing warm_up procedure
        else: Starts at 1 * base_weight then decrease 5/1000 of max weight until it reaches base_weight / 5
        Should be called at each increment counter step!
        Returns:
            -
        """
        # TanH warm-up phase
        if self.counter <= self.kld_warm_up:
            self.weight_kld_n = self._tanh_annealing(self.counter, self.base_weight_kld_n,
                                                     self.kld_tanh_scale, self.kld_warm_up,
                                                     shift=None)
            # using hard-coded parameters for the KLD_z annealing
            self.weight_kld_z = self.base_weight_kld_z  # self._tanh_annealing(self.counter, self.base_weight_kld_z,
            #       0.8, 50)
        # "flat" phase : No need to update
        if self.kld_warm_up < self.counter <= (self.kld_warm_up + self.flat_phase):
            self.weight_kld_n = self.base_weight_kld_n
            self.weight_kld_z = self.base_weight_kld_z
        # Start to decrease weight once counter > warm_up+flat phase for KLD_N
        # No decrease for KLD_Z
        elif self.counter > self.kld_warm_up + self.flat_phase:
            self.weight_kld_n = max(
                self.base_weight_kld_n - (
                        self.kld_decrease * self.base_weight_kld_n * (
                        self.counter - (self.kld_warm_up + self.flat_phase))),
                self.base_weight_kld_n / 4)

    @staticmethod
    def _tanh_annealing(counter, base_weight, scale, warm_up, shift=None):
        """
        epoch_shift sets the epoch at which weight==weight/2, should be set at warm_up//2
        Only updates self.weight_kld as current weight, doesn't return anything
        Computes a 1-shifted (y-axis) tanh and 2/3 * self.warm_up shifted (x-axis) to compute the weight
        using 1+tanh(speed * (epoch - shift))
        Args:
            base_weight: base total weight
            scale: ramp-up speed ~ range [0.05, 0.5]
            epoch_shift:

        Returns:
            weight: kld_weight_n at current epoch
        """

        shift = 2 * warm_up / 3 if shift is None else shift
        return base_weight * (
                1 + math.tanh(scale * (counter - shift))) / 2
        #
        # return self.base_weight_kld_n * (
        #         1 + math.tanh(self.kld_tanh_scale * (self.counter - 2 * self.kld_warm_up / 3))) / 2

    @override
    def increment_counter(self):
        super(MultimodalLossParent, self).increment_counter()
        self._kld_weight_regime()


class JMVAELoss(MultimodalLossParent):
    def __init__(self, max_len_a1=7, max_len_a2=8, max_len_a3=22, max_len_b1=6, max_len_b2=7, max_len_b3=23,
                 max_len_pep=12, aa_dim=20, add_positional_encoding=False, positional_weighting=True,
                 sequence_criterion=nn.MSELoss(reduction='none'), weight_seq=1, weight_kld_z=1, weight_kld_n=1e-2,
                 add_kld_n_marg=False, kld_warm_up=100, kld_tanh_scale=0.1, kld_decrease=5e-3, flat_phase=None,
                 debug=False):
        super(JMVAELoss, self).__init__(sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]),
                                        max_len_pep, aa_dim, add_positional_encoding, positional_weighting,
                                        sequence_criterion,
                                        weight_seq, weight_kld_z, weight_kld_n, kld_warm_up, kld_tanh_scale,
                                        kld_decrease, flat_phase, debug)

        # Using none reduction to apply weight per chain
        assert sequence_criterion.reduction == 'none', f'Reduction mode should be "none" for sequence criterion! Got {sequence_criterion.reduction} instead.'
        self.add_kld_n_marg = add_kld_n_marg
        # Override defining dimensions for vectors
        self.max_len_tcr = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3])
        self.max_len_pep = max_len_pep
        self.aa_dim = aa_dim
        # Override Dimension for positional weighting if applicable
        self.pos_dim_tcr = sum([int(mlx) > 0 for mlx in
                                [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]]) \
            if add_positional_encoding else 0
        self.add_positional_encoding = add_positional_encoding
        # Override positional weighting to set weights for CDR3 to 3x while CDR1,2 weights are at 1
        self.positional_weighting = positional_weighting
        if positional_weighting:
            # take 2d weights (sum_lens, 1), and broadcast using torch.mul so it takes less memory
            alpha_weights = torch.cat([torch.full([max_len_a1 + max_len_a2, 1], 1),
                                       torch.full([max_len_a3, 1], 3)], dim=0)
            beta_weights = torch.cat([torch.full([max_len_b1 + max_len_b2, 1], 1),
                                      torch.full([max_len_b3, 1], 3)], dim=0)
            self.positional_weights = torch.cat([alpha_weights, beta_weights], dim=0).to(self.device)
            assert self.positional_weights.shape == (self.max_len_tcr, 1), 'wrong shape for pos weights'

    def forward(self, trues: List[torch.Tensor], recons: Dict[str, torch.Tensor],
                mus: Dict[str, torch.Tensor], logvars: Dict[str, torch.Tensor]) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # here, trues should be a list containing [x_tcr, x_pep] --> TCR = trues[0], pep = trues[1]
        recon_tcr_marg = self.reconstruction_loss(recons['tcr_marg'], trues[0], which='tcr')
        recon_tcr_joint = self.reconstruction_loss(recons['tcr_joint'], trues[0], which='tcr')
        recon_pep_joint = self.reconstruction_loss(recons['pep_joint'], trues[1], which='pep')
        recon_pep_marg = self.reconstruction_loss(recons['pep_marg'], trues[1], which='pep')

        # Latent KLD ("side" losses)
        kld_tcr_latent = self.kld_latent(mus['tcr_marg'], logvars['tcr_marg'],
                                         mus['joint'], logvars['joint'])
        kld_pep_latent = self.kld_latent(mus['pep_marg'], logvars['pep_marg'],
                                         mus['joint'], logvars['joint'])
        kld_tcr_normal = self.kld_normal(mus['tcr_marg'], logvars['tcr_marg']) if self.add_kld_n_marg else torch.tensor(
            [0.], device=mus['joint'].device,
            requires_grad=True)
        kld_pep_normal = self.kld_normal(mus['pep_marg'], logvars['pep_marg']) if self.add_kld_n_marg else torch.tensor(
            [0.], device=mus['joint'].device,
            requires_grad=True)
        # Normal KLD ("center" loss)
        kld_joint_normal = self.kld_normal(mus['joint'], logvars['joint'])

        if self.debug:
            print('\n', '#' * 15, ' DEBUG ', '#' * 15)
            print('recon_tcr_marg\t', f'{recon_tcr_marg.item():.4f}')
            print('recon_tcr_joint\t', f'{recon_tcr_joint.item():.4f}')
            print('recon_pep_marg\t', f'{recon_pep_marg.item():.4f}')
            print('recon_pep_joint\t', f'{recon_pep_joint.item():.4f}')
            print('kld_tcr_latent\t', f'{kld_tcr_latent.item():.4f}')
            print('kld_pep_latent\t', f'{kld_pep_latent.item():.4f}')
            print('kld_joint_normal\t', f'{kld_joint_normal.item():.4f}')
            print('kld_tcr_normal\t', f'{kld_tcr_normal.item():.4f}')
            print('kld_pep_normal\t', f'{kld_pep_normal.item():.4f}')
            print(self.weight_seq, self.weight_kld_n, self.weight_kld_z, self.positional_weighting)

        recon_loss_marg = {'tcr_marg': recon_tcr_marg, 'pep_marg': recon_pep_marg}
        recon_loss_joint = {'tcr_joint': recon_tcr_joint, 'pep_joint': recon_pep_joint}
        kld_loss_normal = {'tcr_normal': kld_tcr_normal, 'pep_normal': kld_pep_normal, 'joint_normal': kld_joint_normal}
        kld_loss_latent = {'tcr_latent': kld_tcr_latent, 'pep_latent': kld_pep_latent}
        return recon_loss_marg, recon_loss_joint, kld_loss_normal, kld_loss_latent


class BSSVAELoss(MultimodalLossParent):
    def __init__(self, max_len_a1=7, max_len_a2=8, max_len_a3=22, max_len_b1=6, max_len_b2=7, max_len_b3=23,
                 max_len_pep=12, aa_dim=20, add_positional_encoding=False, positional_weighting=True,
                 sequence_criterion=nn.MSELoss(reduction='none'), weight_seq=1, weight_kld_z=1, weight_kld_n=1e-2,
                 add_kld_n_joint=False, kld_warm_up=100, kld_tanh_scale=0.1, kld_decrease=5e-3, flat_phase=None,
                 debug=False):
        # TODO: Define pep_weighted?
        super(BSSVAELoss, self).__init__(sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]),
                                         max_len_pep, aa_dim, add_positional_encoding, positional_weighting,
                                         sequence_criterion,
                                         weight_seq, weight_kld_z, weight_kld_n, kld_warm_up, kld_tanh_scale,
                                         kld_decrease, flat_phase, debug)
        # Using none reduction to apply weight per chain
        assert sequence_criterion.reduction == 'none', f'Reduction mode should be "none" for sequence criterion! Got {sequence_criterion.reduction} instead.'
        self.add_kld_n_joint = add_kld_n_joint
        # Defining dimensions for vectors
        self.max_len_tcr = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3])
        self.max_len_pep = max_len_pep
        self.aa_dim = aa_dim
        # Dimension for positional weighting if applicable
        self.pos_dim_tcr = sum([int(mlx) > 0 for mlx in
                                [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3]]) \
            if add_positional_encoding else 0
        self.add_positional_encoding = add_positional_encoding
        # Reconstruction part
        self.weight_seq = weight_seq
        self.sequence_criterion = sequence_criterion
        # Positional weighting to set weights for CDR3 to 3x while CDR1,2 weights are at 1
        self.positional_weighting = positional_weighting
        if positional_weighting:
            # take 2d weights (sum_lens, 1), and broadcast using torch.mul so it takes less memory
            alpha_weights = torch.cat([torch.full([max_len_a1 + max_len_a2, 1], 1),
                                       torch.full([max_len_a3, 1], 3)], dim=0)
            beta_weights = torch.cat([torch.full([max_len_b1 + max_len_b2, 1], 1),
                                      torch.full([max_len_b3, 1], 3)], dim=0)
            self.positional_weights = torch.cat([alpha_weights, beta_weights], dim=0).to(self.device)
            assert self.positional_weights.shape == (self.max_len_tcr, 1), 'wrong shape for pos weights'

        # # KLD part
        # # Used to do the KLD between Z_marg and Z_joint ; I guess this part should't need a warm-up
        # self.base_weight_kld_z = weight_kld_z
        # # Used to do the KLD between Z_marg and N(0,1) & Parameters for KLD warm-up and annealing
        # self.base_weight_kld_n = weight_kld_n
        # self.add_kld_n_joint = add_kld_n_joint
        # self.weight_kld_n = 0
        # self.weight_kld_z = 1
        # self.kld_tanh_scale = kld_tanh_scale
        # self.kld_warm_up = kld_warm_up
        # self.kld_decrease = kld_decrease
        # self.flat_phase = kld_warm_up // 3 if flat_phase is None else flat_phase
        # self.debug = debug

    def forward(self, trues: List[torch.Tensor], recons: Dict[str, torch.Tensor],
                mus: Dict[str, torch.Tensor], logvars: Dict[str, torch.Tensor]) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        List orders should follow the order of the graphs (left to right)
        TODO: Define pep_weighted?
        Args:
            trues: batch_list of the true tensors. Uses a list because batch is returned as list from dataloader
                    assumes order : trues = [recon_tcr_marg, recon_tcr_joint, recon_pep_joint, recon_pep_marg] (left to right)
            recons: dict of the reconstructed tensors. Same order ^
                                    recons = {'tcr_marg': recon_tcr_marg,
                                              'tcr_joint': recon_tcr_joint,
                                              'pep_joint': recon_pep_joint,
                                              'pep_marg': recon_pep_marg}
            mus: dict of  the mu latent tensors.
                    assumes order : {'tcr_marg': mu_tcr_poe_marg,
                                     'joint': mu_joint_poe,
                                     'pep_marg': mu_pep_poe_marg} (left to right in graph)
            logvars: dict of the logvar latent tensors. Same order & keys ^

        Returns: List of losses, grouped by marginal and joint for each
            recon_marg_losses (dict): dict of recons loss tensors for marginal modalities
                                    order: [recon_tcr_marg, recon_pep_marg]
            recon_joint_losses (dict): dict of recons loss tensors for joint modalities
                                     order: [recon_tcr_joint, recon_pep_joint]
            kld_losses_normal (dict): dict of KLD (normal) loss tensors, i.e. Marginal
                               order: [kld_tcr_normal kld_pep_normal]
            kld_losses_joint (dict): dict of KLD (latent, i.e. joint) loss tensors, i.e. Joint.
                               order: [kld_tcr_joint, kld_pep_joint]
        """
        # Arbitrarily upweight TCR loss 3:1
        recon_tcr_marg = 3 * self.reconstruction_loss(recons['tcr_marg'], trues[0], which='tcr')
        recon_tcr_joint = 3 * self.reconstruction_loss(recons['tcr_joint'], trues[1], which='tcr')
        recon_pep_joint = self.reconstruction_loss(recons['pep_joint'], trues[2], which='pep')
        recon_pep_marg = self.reconstruction_loss(recons['pep_marg'], trues[3], which='pep')

        kld_tcr_normal = self.kld_normal(mus['tcr_marg'], logvars['tcr_marg'])
        kld_tcr_joint = self.kld_latent(mus['tcr_marg'], logvars['tcr_marg'], mus['joint'], logvars['joint'])
        kld_pep_joint = self.kld_latent(mus['pep_marg'], logvars['pep_marg'], mus['joint'], logvars['joint'])
        kld_pep_normal = self.kld_normal(mus['pep_marg'], logvars['pep_marg'])

        kld_n_joint = self.kld_normal(mus['joint'], logvars['joint']) if self.add_kld_n_joint else torch.tensor([0.],
                                                                                                                device=
                                                                                                                mus[
                                                                                                                    'joint'].device,
                                                                                                                requires_grad=True)

        if self.debug:
            print('\n', '#' * 15, ' DEBUG ', '#' * 15)
            print('recon_tcr_marg\t', f'{recon_tcr_marg.item():.4f}')
            print('recon_tcr_joint\t', f'{recon_tcr_joint.item():.4f}')
            print('recon_pep_marg\t', f'{recon_pep_marg.item():.4f}')
            print('recon_pep_joint\t', f'{recon_pep_joint.item():.4f}')
            print('kld_tcr_normal\t', f'{kld_tcr_normal.item():.4f}')
            print('kld_pep_normal\t', f'{kld_pep_normal.item():.4f}')
            print('kld_joint_normal\t', f'{kld_n_joint.item():.4f}')
            print('kld_tcr_joint\t', f'{kld_tcr_joint.item():.4f}')
            print('kld_pep_joint\t', f'{kld_pep_joint.item():.4f}')
            print(self.weight_seq, self.weight_kld_n, self.weight_kld_z, self.positional_weighting)

        recon_loss_marg = {'tcr_marg': recon_tcr_marg, 'pep_marg': recon_pep_marg}
        recon_loss_joint = {'tcr_joint': recon_tcr_joint, 'pep_joint': recon_pep_joint}
        kld_loss_normal = {'tcr_normal': kld_tcr_normal, 'pep_normal': kld_pep_normal, 'joint_normal': kld_n_joint}
        kld_loss_latent = {'tcr_joint': kld_tcr_joint, 'pep_joint': kld_pep_joint}

        return recon_loss_marg, recon_loss_joint, kld_loss_normal, kld_loss_latent

    # TODO : Delete this instead of commenting
    # def reconstruction_loss(self, x_hat, x, which):
    #     """
    #     Uses slicing to reshape the vectors and apply the reconstruction loss
    #     Args:
    #         x_hat:
    #         x:
    #         which: Defines which vector we are slicing-reshaping. Should be 'pep' or 'tcr'
    #
    #     Returns: mean loss for current batch x, x_hat
    #     """
    #     x_hat_seq, positional_hat = self.slice_x(x_hat, which)
    #     x_true_seq, positional_true = self.slice_x(x, which)
    #     reconstruction_loss = self.sequence_criterion(x_hat_seq, x_true_seq)
    #     # if positional weighting, then multiply the loss to give larger/smaller gradients w.r.t. chains and positions
    #     if self.positional_weighting and which == 'tcr':
    #         reconstruction_loss = reconstruction_loss.mul(self.positional_weights)
    #
    #     # Here, take the mean before checking for positional encoding because we have un-reduced loss
    #     # and scale it by the "weight_seq" (versus weight_kld)
    #     reconstruction_loss = self.weight_seq * reconstruction_loss.mean()
    #
    #     # check if we use positional encoding, if yes we add the loss
    #     if self.add_positional_encoding:
    #         # Here use 5e-4, very small weight because this task should be trivial
    #         positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true)
    #         reconstruction_loss += (5e-4 * self.weight_seq * positional_loss)
    #
    #     return reconstruction_loss
    #
    # def kld_latent(self, mu_1, logvar_1, mu_2, logvar_2):
    #     """
    #     Compute the Kullback-Leibler Divergence between two Gaussian distributions.
    #
    #     Parameters:
    #     - mu1: Mean of the first distribution.
    #     - logvar1: Log variance of the first distribution.
    #     - mu2: Mean of the second distribution.
    #     - logvar2: Log variance of the second distribution.
    #
    #     Returns:
    #     - The KLD from Q to P.
    #     """
    #     sigma1_squared = logvar_1.exp()
    #     sigma2_squared = logvar_2.exp()
    #     kld = 0.5 * torch.sum(
    #         logvar_1 - logvar_2 - 1 + sigma2_squared / sigma1_squared + (mu_2 - mu_1).pow(2) / sigma1_squared, dim=-1)
    #     return self.weight_kld_z * kld.mean()
    #
    # def kld_normal(self, mu, logvar):
    #     kld = 1 + logvar - mu.pow(2) - logvar.exp()
    #     return self.weight_kld_n * (-0.5 * torch.mean(kld))
    #
    # # class utils methods
    # def slice_x(self, x, which):
    #     if which == 'tcr':
    #         sequence = x[:, 0:self.max_len_tcr * (self.aa_dim + self.pos_dim_tcr)].view(-1, self.max_len_tcr,
    #                                                                                     self.aa_dim + self.pos_dim_tcr)
    #     elif which == 'pep':
    #         sequence = x[:, 0:self.max_len_pep * self.aa_dim].view(-1, self.max_len_pep, self.aa_dim)
    #     positional_encoding = None
    #     if self.add_positional_encoding:
    #         positional_encoding = sequence[:, :, self.aa_dim:]
    #         sequence = sequence[:, :, :self.aa_dim]
    #     return sequence, positional_encoding
    #
    # def _kld_weight_regime(self):
    #     """
    #     if counter <= warm_up : tanh annealing warm_up procedure
    #     else: Starts at 1 * base_weight then decrease 5/1000 of max weight until it reaches base_weight / 5
    #     Should be called at each increment counter step!
    #     Returns:
    #         -
    #     """
    #     # TanH warm-up phase
    #     if self.counter <= self.kld_warm_up:
    #         self.weight_kld_n = self._tanh_annealing(self.counter, self.base_weight_kld_n,
    #                                                  self.kld_tanh_scale, self.kld_warm_up,
    #                                                  shift=None)
    #         # using hard-coded parameters for the KLD_z annealing
    #         self.weight_kld_z = self.base_weight_kld_z  # self._tanh_annealing(self.counter, self.base_weight_kld_z,
    #         #       0.8, 50)
    #     # "flat" phase : No need to update
    #     if self.kld_warm_up < self.counter <= (self.kld_warm_up + self.flat_phase):
    #         self.weight_kld_n = self.base_weight_kld_n
    #         self.weight_kld_z = self.base_weight_kld_z
    #     # Start to decrease weight once counter > warm_up+flat phase for KLD_N
    #     # No decrease for KLD_Z
    #     elif self.counter > self.kld_warm_up + self.flat_phase:
    #         self.weight_kld_n = max(
    #             self.base_weight_kld_n - (
    #                     self.kld_decrease * self.base_weight_kld_n * (
    #                     self.counter - (self.kld_warm_up + self.flat_phase))),
    #             self.base_weight_kld_n / 4)
    #
    # @staticmethod
    # def _tanh_annealing(counter, base_weight, scale, warm_up, shift=None):
    #     """
    #     epoch_shift sets the epoch at which weight==weight/2, should be set at warm_up//2
    #     Only updates self.weight_kld as current weight, doesn't return anything
    #     Computes a 1-shifted (y-axis) tanh and 2/3 * self.warm_up shifted (x-axis) to compute the weight
    #     using 1+tanh(speed * (epoch - shift))
    #     Args:
    #         base_weight: base total weight
    #         scale: ramp-up speed ~ range [0.05, 0.5]
    #         epoch_shift:
    #
    #     Returns:
    #         weight: kld_weight_n at current epoch
    #     """
    #
    #     shift = 2 * warm_up / 3 if shift is None else shift
    #     return base_weight * (
    #             1 + math.tanh(scale * (counter - shift))) / 2
    #     #
    #     # return self.base_weight_kld_n * (
    #     #         1 + math.tanh(self.kld_tanh_scale * (self.counter - 2 * self.kld_warm_up / 3))) / 2
    #
    # @override
    # def increment_counter(self):
    #     super(BSSVAELoss, self).increment_counter()
    #     self._kld_weight_regime()
