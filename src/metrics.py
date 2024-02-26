import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from src.torch_utils import mask_modality
from torch import nn
from torch.nn import functional as F

mpl.rcParams['figure.dpi'] = 180
sns.set_style('darkgrid')
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, \
    recall_score, precision_score, precision_recall_curve, auc, average_precision_score


class LossParent(nn.Module):

    def __init__(self, debug=False):
        super(LossParent, self).__init__()
        self.counter = 0
        self.debug = debug

    def increment_counter(self):
        "one level of children modules ; If we get too many might need to have a recursive method somewhere"
        self.counter += 1
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'increment_counter'):
                c.increment_counter()


class VAELoss(LossParent):
    """
    No fucking annealing, just some basic stuff for now
    TODO: re-do the tanh behaviour for the KLD loss
    Does the VAE part of the loss, including reconstruction error (here for the sequence and possible positional encoding vectors)
         and the KLD to control the latent dimensions

    Inherits from the LossParent class and has a built-in counter
    counter increment should be called externally, e.g. 1x per epoch, in train-eval loop functions

    **17.01.24 : warm_up counter should now be an epoch counter and not batch counter**
    --> All I need to ensure : FullTCRvae works, fulltcrvae_triplet works, bimodal works
    """

    def __init__(self, sequence_criterion=nn.MSELoss(reduction='mean'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 weight_seq=3, weight_kld=1e-2, debug=False, warm_up=0):
        super(VAELoss, self).__init__()
        max_len = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])
        pos_dim = sum([int(mlx) > 0 for mlx in
                       [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep]]) \
            if add_positional_encoding else 0
        self.max_len = max_len
        self.aa_dim = aa_dim
        self.pos_dim = pos_dim
        self.add_positional_encoding = add_positional_encoding
        self.norm_factor = weight_seq + weight_kld
        self.sequence_criterion = sequence_criterion
        self.weight_seq = weight_seq / self.norm_factor
        self.base_weight_kld = weight_kld / self.norm_factor
        self.weight_kld = self.base_weight_kld
        self.step = 0
        self.debug = debug
        self.kld_warm_up = warm_up
        print(f'Weights: seq, kld_base: ', self.weight_seq, self.base_weight_kld)

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

    def forward(self, x_hat, x, mu, logvar):
        """
        x_hat and x should be the flattened vectors, where hat is the reconstructed and x the true
        Then, self.slice_x will deal with slicing, reshaping and (if appl.) extracting the positional encoding
        KLD weight follows 2 regimes:
            1: Warm-Up period, w_kld = 0 until counter==warm_up
            2: w_kld = w_base_kld, then decreases each epoch by a set amount until it reaches a min of w_base / 5
        Args:
            x_hat:
            x:
            mu:
            logvar:

        Returns:

        """
        x_hat_seq, positional_hat = self.slice_x(x_hat)
        x_true_seq, positional_true = self.slice_x(x)
        reconstruction_loss = self.weight_seq * self.sequence_criterion(x_hat_seq, x_true_seq)

        if self.add_positional_encoding:
            # TODO Should maybe add a weight here to minimize this part because it's easy
            #      For now, try to use weight = 1/100 weight_seq ?
            # Here use 1e-4
            positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true)
            reconstruction_loss += (1e-4 * self.weight_seq * positional_loss)

        # KLD weight regime control
        if self.counter < self.kld_warm_up:
            # While in the warm-up phase, weight_kld is 0
            self.weight_kld = 0
        else:
            # Otherwise, it starts at the base_kld weight and decreases a 1% of max weight with the epoch counter
            # until it reaches a minimum of the base weight / 5
            self.weight_kld = max(
                self.base_weight_kld - (0.01 * self.base_weight_kld * (self.counter - self.kld_warm_up)),
                self.base_weight_kld / 5)

        kld = self.weight_kld * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

        if self.debug:
            print('seq_loss', reconstruction_loss)
            print('kld_weight', self.weight_kld)
            print('kld_loss', kld, '\n')

        # Return them separately and sum later so that I can debug each component
        return reconstruction_loss, kld

    def set_debug(self, debug):
        self.debug = debug

    def reset_parameters(self):
        self.step = 0


class TripletLoss(LossParent):
    def __init__(self, dist_type='cosine', margin=None):
        super(TripletLoss, self).__init__()
        assert dist_type in ['cosine', 'l1',
                             'l2'], f'Distance type must be in ["l1", "l2", "cosine"]! Got {dist_type} instead.'
        self.distance = {'cosine': compute_cosine_distance, 'l2': torch.cdist, 'l1': torch.cdist}[dist_type]
        self.p = 1 if dist_type == 'l1' else 2 if dist_type == 'l2' else None
        if margin is None:
            margin = 0.25 if dist_type == 'cosine' else 1.0
        self.margin = margin

    def forward(self, z, labels, **kwargs):
        weights = kwargs['pep_weights'] if (
                'pep_weights' in kwargs and kwargs['pep_weights'] is not None) else torch.ones([len(z)])
        weights = weights.to(z.device)
        pairwise_distances = self.distance(z, z, p=self.p)
        mask_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask_negative = 1 - mask_positive
        # Get the distances for each of the masks
        positive_distances = mask_positive * pairwise_distances
        negative_distances = mask_negative * pairwise_distances
        # Take the relu to encourage error above margin (i.e. threshold)
        loss = torch.nn.functional.relu(positive_distances - negative_distances + self.margin) * weights
        loss = loss.mean()
        return loss


class CombinedVAELoss(LossParent):
    """
    This is the VAE + Triplet Loss, that is used kind of everywhere now
    """

    def __init__(self, sequence_criterion=nn.MSELoss(reduction='mean'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 weight_seq=1, weight_kld=1e-2, debug=False, warm_up=10, weight_vae=1, weight_triplet=1,
                 dist_type='cosine', triplet_loss_margin=None):
        super(CombinedVAELoss, self).__init__()
        # TODO: PHASE OUT N BATCHES
        self.vae_loss = VAELoss(sequence_criterion, aa_dim=aa_dim, max_len_a1=max_len_a1, max_len_a2=max_len_a2,
                                max_len_a3=max_len_a3, max_len_b1=max_len_b1, max_len_b2=max_len_b2,
                                max_len_b3=max_len_b3, max_len_pep=max_len_pep,
                                add_positional_encoding=add_positional_encoding, weight_seq=weight_seq,
                                weight_kld=weight_kld, debug=debug, warm_up=warm_up)
        self.triplet_loss = TripletLoss(dist_type, triplet_loss_margin)
        self.weight_triplet = weight_triplet
        self.weight_vae = weight_vae
        self.norm_factor = (self.weight_triplet + self.weight_vae)
        print(f'weights: vae, triplet: ', self.weight_vae, self.weight_triplet)

    # Maybe this normalisation factor thing is not needed
    def forward(self, x_hat, x, mu, logvar, z, labels, **kwargs):
        # Both the vaeloss and triplet loss are mean reduced already ; Just need to adjust weights for each term (?)
        recon_loss, kld_loss = self.vae_loss(x_hat, x, mu, logvar)
        recon_loss = self.weight_vae * recon_loss / self.norm_factor
        kld_loss = self.weight_vae * kld_loss / self.norm_factor
        triplet_loss = self.weight_triplet * self.triplet_loss(z, labels, **kwargs) / self.norm_factor
        return recon_loss, kld_loss, triplet_loss


class TwoStageVAELoss(LossParent):
    # Here
    def __init__(self, sequence_criterion=nn.MSELoss(reduction='mean'), aa_dim=20, max_len_a1=0, max_len_a2=0,
                 max_len_a3=22, max_len_b1=0, max_len_b2=0, max_len_b3=23, max_len_pep=0, add_positional_encoding=False,
                 weight_seq=3, weight_kld=1, debug=False, warm_up=0, warm_up_clf=0, weight_vae=1, weight_triplet=1,
                 weight_classification=1, dist_type='cosine', triplet_loss_margin=None):
        super(TwoStageVAELoss, self).__init__()
        # TODO: Here, maybe change the additional term from triplet loss to something else (Center Loss or Contrastive loss?)
        self.vae_loss = VAELoss(sequence_criterion, aa_dim=aa_dim, max_len_a1=max_len_a1, max_len_a2=max_len_a2,
                                max_len_a3=max_len_a3, max_len_b1=max_len_b1, max_len_b2=max_len_b2,
                                max_len_b3=max_len_b3, max_len_pep=max_len_pep,
                                add_positional_encoding=add_positional_encoding, weight_seq=weight_seq,
                                weight_kld=weight_kld, debug=debug, warm_up=warm_up)
        self.triplet_loss = TripletLoss(dist_type, triplet_loss_margin)
        self.classification_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.weight_triplet = float(weight_triplet)
        self.weight_vae = float(weight_vae)
        self.weight_classification = float(weight_classification)
        self.norm_factor = (self.weight_triplet + self.weight_vae + self.weight_classification)
        self.warm_up_clf = warm_up_clf
        self.counter = 0

    # Maybe this normalisation factor thing is not needed
    def forward(self, x_hat, x, mu, logvar, z, triplet_labels, x_out, binder_labels, **kwargs):
        weights = kwargs['pep_weights'] if (
                'pep_weights' in kwargs and kwargs['pep_weights'] is not None) else torch.ones([len(z)])
        weights = weights.to(x.device)
        # Both the vaeloss and triplet loss are mean reduced already ; Just need to adjust weights for each term (?)
        recon_loss, kld_loss = self.vae_loss(x_hat, x, mu, logvar)
        recon_loss = self.weight_vae * recon_loss / self.norm_factor
        kld_loss = self.weight_vae * kld_loss / self.norm_factor
        triplet_loss = self.weight_triplet * self.triplet_loss(z, triplet_labels,
                                                               pep_weights=weights) / self.norm_factor

        if self.counter >= self.warm_up_clf:
            # Here criterion should be with reduction='none' and then manually do the mean() because `weight` is not used in forward but init
            # but now I don't even use necessarily use sample weights
            # TODO : Here, removed "* weights" from the classification loss before the mean() because we are using weights to remove triplet from swapped datapoints
            #       i.e. : Triplet is only trained with positive points, whereas the rest are trained with all the losses
            classification_loss = (self.weight_classification * self.classification_loss(x_out,
                                                                                         binder_labels)).mean() / self.norm_factor
            if self.debug:
                print('clf pep weights, loss', weights[:10], classification_loss)
        else:
            classification_loss = torch.tensor([0])
        return recon_loss, kld_loss, triplet_loss, classification_loss

    def increment_counter(self):
        self.counter += 1


class TrimodalVAELoss(LossParent):
    """
        Disabling some elements' loss term should be done here by using the masks
        Keep the VAE model itself simple to only always return all modalities, unmasked
    """

    def __init__(self, alpha_dim, beta_dim, pep_dim, sequence_criterion=nn.MSELoss(reduction='none'),
                 aa_dim=20, add_positional_encoding=False,
                 weight_seq=1, weight_kld=1e-2, weight_triplet=0, weight_vae=1,
                 dist_type='cosine', triplet_loss_margin=0.075,
                 debug=False, warm_up=15):
        super(TrimodalVAELoss, self).__init__()
        assert sequence_criterion.reduction == 'none', f'Reduction mode should be "none" for sequence criterion! Got {sequence_criterion.reduction} instead.'
        self.sequence_criterion = sequence_criterion
        self.alpha_dim = alpha_dim
        self.beta_dim = beta_dim
        self.pep_dim = pep_dim
        self.aa_dim = aa_dim
        self.pos_dim = 1 if add_positional_encoding else 0
        self.add_positional_encoding = add_positional_encoding
        self.norm_factor = weight_seq + weight_kld + weight_triplet
        self.weight_seq = weight_seq
        self.base_weight_kld = weight_kld
        self.weight_triplet = weight_triplet
        self.weight_kld = self.base_weight_kld
        self.weight_vae = weight_vae

        self.kld_warm_up = warm_up
        self.triplet_loss = TripletLoss(dist_type, triplet_loss_margin)
        self.norm_factor = (self.weight_triplet + self.weight_vae)
        self.debug = debug
        print(f'weights: vae, triplet: ', self.weight_vae, self.weight_triplet)

    def forward(self, x_hat_alpha, x_true_alpha, x_hat_beta, x_true_beta, x_hat_pep, x_true_pep,
                mu_joint, logvar_joint, mus: list, logvars: list, mask_alpha, mask_beta, mask_pep,
                triplet_labels=None, **kwargs):
        """
        Uses the masks to remove missing modalities for each loss term.
        This masking behaviour is taken care of in the various functions doing the losses instead
        of being explicitely laid out in the forward here
        Args:
            x_hat_alpha:
            x_true_alpha:
            x_hat_beta:
            x_true_beta:
            x_hat_pep:
            x_true_pep:
            mu_joint:
            logvar_joint:
            mus:
            logvars:
            mask_alpha:
            mask_beta:
            mask_pep:
            triplet_labels:
            **kwargs:

        Returns:

        """
        # Reconstruction losses ; Put more weight on alpha and beta because longer seq
        recon_loss_alpha = 3 * self.reconstruction_loss(x_hat_alpha, x_true_alpha, 'alpha', mask=mask_alpha)
        recon_loss_beta = 3 * self.reconstruction_loss(x_hat_beta, x_true_beta, 'beta', mask=mask_beta)
        recon_loss_pep = self.reconstruction_loss(x_hat_pep, x_true_pep, 'pep', mask=mask_pep)
        reconstruction_loss = (recon_loss_alpha + recon_loss_beta + recon_loss_pep) / 7

        # KLD weight regime control ; Handles both the joint and marginal KLDs
        if self.counter < self.kld_warm_up:
            # While in the warm-up phase, weight_kld is 0
            self.weight_kld = 0
        else:
            # Otherwise, it starts at the base_kld weight and decreases a 1% of max weight with the epoch counter
            # until it reaches a minimum of the base weight / 5
            self.weight_kld = max(
                self.base_weight_kld - (0.01 * self.base_weight_kld * (self.counter - self.kld_warm_up)),
                self.base_weight_kld / 5)
        kld_joint = self.kullback_leibler_divergence(mu_joint, logvar_joint, mask=None)
        kld_alpha = self.kullback_leibler_divergence(mus[0], logvars[0], mask=mask_alpha)
        kld_beta = self.kullback_leibler_divergence(mus[1], logvars[1], mask=mask_beta)
        kld_pep = self.kullback_leibler_divergence(mus[2], logvars[2], mask=mask_pep)
        kld_marginal = kld_alpha + kld_beta + kld_pep
        if self.debug:
            print('counter', self.counter)
            print('seq_loss', reconstruction_loss)
            print('kld_weight', self.weight_kld)
            print('kld_loss', kld_joint.round(decimals=4), kld_alpha.round(decimals=4), kld_beta.round(decimals=4), kld_pep.round(decimals=4), '\n')

        # This here should probably be changed : Triplet labels wouldn't exist for all of mu_joint
        # Because of the tri-modality, technically we could have datapoints with missing peptide input (?)
        triplet_loss = self.weight_triplet * self.triplet_loss(mu_joint, triplet_labels, **kwargs)
        # Return them separately and sum later so that I can debug each component
        return reconstruction_loss / self.norm_factor, kld_joint / self.norm_factor, kld_marginal / self.norm_factor, triplet_loss / self.norm_factor

    def kullback_leibler_divergence(self, mu, logvar, mask=None):
        kld = 1 + logvar - mu.pow(2) - logvar.exp()
        if mask is not None:
            kld = mask_modality(kld, mask)
        return self.weight_kld * (-0.5 * torch.mean(kld))

    def reconstruction_loss(self, x_hat, x_true, which, mask):
        x_hat_seq, positional_hat = self.slice_x(x_hat, which)
        x_true_seq, positional_true = self.slice_x(x_true, which)
        reconstruction_loss = self.weight_seq * self.sequence_criterion(x_hat_seq, x_true_seq)
        reconstruction_loss = torch.nanmean(mask_modality(reconstruction_loss, mask, np.nan))
        if self.add_positional_encoding:
            positional_loss = F.binary_cross_entropy_with_logits(positional_hat, positional_true, reduction='none')
            positional_loss = torch.mean(mask_modality(positional_loss, mask, np.nan))
            reconstruction_loss += (1e-4 * self.weight_seq * positional_loss)
        return reconstruction_loss

    def slice_x(self, x_flat, which: str):
        """
        Slices and extracts // reshapes the sequence vector from a flattened tensor
        Also extracts the positional encoding vector if used (?)
        Args:
            x:

        Returns:

        """
        # The slicing part exist here as legacy code in case we want to add other features to the end of the vector
        # In this case, we need to first slice the first part of the vector which is the sequence, then additionally
        # slice the rest of the vector which should be whatever feature we add in
        # Here, reshape to aa_dim+pos_dim no matter what, as pos_dim can be 0, and first set pos_enc to None
        max_len = {'alpha': self.alpha_dim,
                   'beta': self.beta_dim,
                   'pep': self.pep_dim}[which]

        sequence_tensor = x_flat[:, 0:(max_len * (self.aa_dim + self.pos_dim))].view(-1, max_len,
                                                                                     (self.aa_dim + self.pos_dim))
        positional_encoding_tensor = None
        # Then, if self.pos_dim is not 0, further slice the tensor to recover each part
        if self.add_positional_encoding:
            positional_encoding_tensor = sequence_tensor[:, :, self.aa_dim:]
            sequence_tensor = sequence_tensor[:, :, :self.aa_dim]
        return sequence_tensor, positional_encoding_tensor


def compute_cosine_distance(z_embedding: torch.Tensor, *args, **kwargs):
    """
    Computes a square cosine distance matrix (All vs ALl) for a given sample of latent Z
    Args:
        z_embedding (torch.Tensor): Latent embedding, dimension N x latent_dim
        *args:
        **kwargs:

    Returns:
        cosine_distance_matrix (torch.Tensor): Square tensor of dimension NxN containing pairwise cosine distances
    """
    # Compute the dot product of the embedding matrix
    dot_product = torch.mm(z_embedding, z_embedding.t())

    # Compute the L2 norms of the vectors
    norms = torch.norm(z_embedding, p=2, dim=1, keepdim=True)

    # Compute the pairwise cosine distances
    cosine_distance_matrix = 1 - (dot_product / (norms * norms.t()))
    # Clamps the negative values to 0
    cosine_distance_matrix[cosine_distance_matrix <= 1e-6] = 0

    return cosine_distance_matrix


def model_reconstruction_stats(model, x_reconstructed, x_true, return_per_element=False, modality_mask=None):
    # Here concat the list in case we are given a list of tensors (as done in the train/valid batching system)
    x_reconstructed = torch.cat(x_reconstructed) if type(x_reconstructed) == list else x_reconstructed
    x_true = torch.cat(x_true) if type(x_true) == list else x_true
    seq_hat, positional_hat = model.reconstruct_hat(x_reconstructed)
    seq_true, positional_true = model.reconstruct_hat(x_true)
    seq_accuracy, positional_accuracy = reconstruction_accuracy(seq_true, seq_hat, positional_true, positional_hat,
                                                                pad_index=20, return_per_element=return_per_element,
                                                                modality_mask=modality_mask)
    metrics = {'seq_accuracy': seq_accuracy, 'pos_accuracy': positional_accuracy}
    if positional_accuracy is None:
        del metrics['pos_accuracy']

    return metrics


# NOTE : Fixed version with the true acc
# TODO : add thing for positional encoding
def reconstruction_accuracy(seq_true, seq_hat, positional_true, positional_hat,
                            pad_index=20, return_per_element=False, modality_mask=None):
    """
    Args:
        mask:
        seq_true: ordinal vector of true sequence (Here, the number is to map back to an amino acid with AA_KEYS, with 20 = X)
        seq_hat: Same but reconstructed
        positional_true: positional encoding (true), shape is either 2D or flattened vector
        positional_hat: positional encoding (reconstructed), same as above
        pad_index: This is the INDEX for the AA used for padding. 20 for default (which represents X)
        return_per_element: Return a per-element acc instead of mean

    Returns:
        seq_accuracy : sequence reconstruction accuracy
                       Either a list of accuracy per sample if return_per_element is True, or a mean number
        pos_accuracy : positional encoding reconstruction accuracy
                       Same as above, except it can be None if positional_hat is None (i.e. disabled)

    """
    # Use modality_mask to do something

    # Compute the mask and true lengths
    mask = (seq_true != pad_index).float()
    true_lens = mask.sum(dim=1)
    # difference here for per element is that we don't take the mean(dim=0) and
    # have to detach() from graph to do tolist() ; Here is still per element
    seq_accuracy = ((seq_true == seq_hat).float() * mask).sum(1) / true_lens
    # Assuming pos_accuracy are logits : sigmoid->threshold(0.5)->compare to true
    pos_accuracy = ((F.sigmoid(positional_hat) > 0.5).float() == positional_true).float().mean(
        dim=(1, 2)) if positional_hat is not None else None
    # fill with nans to ignore those datapoints
    seq_accuracy = mask_modality(seq_accuracy, modality_mask, torch.nan)
    pos_accuracy = mask_modality(pos_accuracy, modality_mask, torch.nan) if positional_hat is not None else None
    if return_per_element:
        # Here give a per element accuracy (so that we have individual metric for each datapoint in a df)
        seq_accuracy = seq_accuracy.detach().cpu().tolist()
        pos_accuracy = pos_accuracy.detach().cpu().tolist() if positional_hat is not None else None
    else:
        # Here gives the mean accuracy of the model for all the datapoints
        seq_accuracy = seq_accuracy.nanmean(dim=0).item()
        pos_accuracy = pos_accuracy.nanmean(dim=0).item() if positional_hat is not None else None

    return seq_accuracy, pos_accuracy


def auc01_score(y_true: np.ndarray, y_pred: np.ndarray, max_fpr=0.1) -> float:
    """Compute the partial AUC of the ROC curve for FPR up to max_fpr.
    Args:
        y_true (array-like): The true labels of the data (0 or 1).
        y_pred (array-like): The predicted probabilities or scores.
        max_fpr (float): Maximum false positive rate.
    Returns:
        float: Partial AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return auc(fpr, tpr) * 10


def get_metrics(y_true, y_score, y_pred=None, threshold=0.50, keep=False, reduced=True, round_digit=4) -> dict:
    """
    Computes all classification metrics & returns a dictionary containing the various key/metrics
    incl. ROC curve, AUC, AUC_01, F1 score, Accuracy, Recall
    Args:
        y_true:
        y_pred:
        y_score:

    Returns:
        metrics (dict): Dictionary containing all results
    """
    metrics = {}
    # DETACH & PASS EVERYTHING TO CPU
    if threshold is not None and y_pred is None:
        # If no y_pred is provided, will threshold score (y in [0, 1])
        y_pred = (y_score > threshold)
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy()
        elif type(y_pred) == np.ndarray:
            y_pred = y_pred.astype(int)
    elif y_pred is not None and type(y_pred) == torch.Tensor:
        y_pred = y_pred.int().cpu().detach().numpy()

    if type(y_true) == torch.Tensor and type(y_score) == torch.Tensor:
        y_true, y_score = y_true.int().cpu().detach().numpy(), y_score.cpu().detach().numpy()

    metrics['auc'] = roc_auc_score(y_true, y_score)
    metrics['auc_01'] = roc_auc_score(y_true, y_score, max_fpr=0.1)
    metrics['auc_01_real'] = auc01_score(y_true, y_score, max_fpr=0.1)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AP'] = average_precision_score(y_true, y_score)
    if not reduced:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics['roc_curve'] = fpr, tpr
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['pr_curve'] = recall, precision  # So it follows the same x,y format as roc_curve
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
            metrics['prauc'] = auc(recall, precision)
            metrics['AP'] = average_precision_score(y_true, y_score)
        except:
            print('Couldn\'t get AUCs/etc because there\'s only one class in the dataset')
            print(f'Only negatives: {all(y_true == 0)}, Only positives: {all(y_true == 1)}')
            raise ValueError
        if keep:
            metrics['y_true'] = y_true
            metrics['y_score'] = y_score
    if round_digit is not None:
        for k, v in metrics.items():
            metrics[k] = round(v, round_digit)
    return metrics


def plot_roc_auc_fold(results_dict, palette='hsv', n_colors=None, fig=None, ax=None,
                      title='ROC AUC plot\nPerformance for average prediction from models of each fold',
                      bbox_to_anchor=(0.9, -0.1)):
    n_colors = len(results_dict.keys()) if n_colors is None else n_colors
    sns.set_palette(palette, n_colors=n_colors)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    print(results_dict.keys())
    for k in results_dict:
        if k == 'kwargs': continue
        fpr = results_dict[k]['roc_curve'][0]
        tpr = results_dict[k]['roc_curve'][1]
        auc = results_dict[k]['auc']
        auc_01 = results_dict[k]['auc_01']
        # print(k, auc, auc_01)
        style = '--' if type(k) == np.int32 else '-'
        alpha = 0.75 if type(k) == np.int32 else .9
        lw = .8 if type(k) == np.int32 else 1.5
        sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'{k}, AUC={auc.round(4)}, AUC_01={auc_01.round(4)}',
                     n_boot=50, ls=style, lw=lw, alpha=alpha)

    sns.lineplot([0, 1], [0, 1], ax=ax, ls='--', color='k', label='random', lw=0.5)
    if bbox_to_anchor is not None:
        ax.legend(bbox_to_anchor=bbox_to_anchor)

    ax.set_title(f'{title}')
    return fig, ax


def get_mean_roc_curve(roc_curves, extra_key=None, auc01=False):
    """
    Assumes a single-level dict, i.e. roc_curves_dict has all the outer folds, and no inner folds
    Or it is the sub-dict that contains all the inner folds for a given outer fold.
    i.e. to access a given fold's curve, should use `roc_curves_dict[number]['roc_curve']`
    Args:
        roc_curves_dict:
        extra_key (str) : Extra_key in case it's nested, like train_metrics[fold]['valid']['roc_curve']
    Returns:
        base_fpr
        mean_curve
        low_std_curve
        high_std_curve
        auc
    """

    # Base fpr to interpolate
    tprs = []
    aucs = []
    aucs_01 = []
    if type(roc_curves) == dict:
        if extra_key is not None:
            max_n = max([len(v[extra_key]['roc_curve'][0]) for k, v in roc_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_fpr = np.linspace(0, 1, max_n)
            for k, v in roc_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                fpr = v[extra_key]['roc_curve'][0]
                tpr = v[extra_key]['roc_curve'][1]
                # Interp TPR so it fits the right shape for base_fpr
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                tprs.append(tpr)
                aucs.append(v[extra_key]['auc'])
        else:
            max_n = max([len(v['roc_curve'][0]) for k, v in roc_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_fpr = np.linspace(0, 1, max_n)

            for k, v in roc_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                fpr = v['roc_curve'][0]
                tpr = v['roc_curve'][1]
                # Interp TPR so it fits the right shape for base_fpr
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                tprs.append(tpr)
                aucs.append(v['auc'])

    elif type(roc_curves) == list:
        # THIS HERE ASSUMES THE RESULTS ARE IN FORMAT [((fpr, tpr), auc) ...]
        max_n = max([len(x[0][0]) for x in roc_curves])
        base_fpr = np.linspace(0, 1, max_n)

        for curves in roc_curves:
            fpr = curves[0][0]
            tpr = curves[0][1]
            # Interp TPR so it fits the right shape for base_fpr
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            tprs.append(tpr)
            aucs.append(curves[1])
            if auc01:
                aucs_01.append(curves[-1])

    mean_auc = np.mean(aucs)
    if auc01:
        mean_auc01 = np.mean(aucs_01)

    tprs = np.stack(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    upper = np.minimum(mean_tprs + std_tprs, 1)
    lower = mean_tprs - std_tprs

    if auc01:
        return base_fpr, mean_tprs, lower, upper, mean_auc, mean_auc01
    else:
        return base_fpr, mean_tprs, lower, upper, mean_auc


def get_mean_pr_curve(pr_curves, extra_key=None):
    """
    Assumes a single-level dict, i.e. roc_curves_dict has all the outer folds, and no inner folds
    Or it is the sub-dict that contains all the inner folds for a given outer fold.
    i.e. to access a given fold's curve, should use `roc_curves_dict[number]['roc_curve']`
    Args:
        roc_curves_dict:
        extra_key (str) : Extra_key in case it's nested, like train_metrics[fold]['valid']['roc_curve']
    Returns:
        base_recall
        mean_curve
        std_curve
    """

    # Base recall to interpolate
    precisions = []
    aucs = []
    if type(pr_curves) == dict:
        if extra_key is not None:
            max_n = max([len(v[extra_key]['pr_curve'][0]) for k, v in pr_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_recall = np.linspace(0, 1, max_n)
            for k, v in pr_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                recall = v[extra_key]['pr_curve'][0]
                precision = v[extra_key]['pr_curve'][1]
                # Interp precision so it fits the right shape for base_recall
                precision = np.interp(base_recall, recall, precision)
                precision[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                precisions.append(precision)
                aucs.append(v[extra_key]['auc'])
        else:
            max_n = max([len(v['pr_curve'][0]) for k, v in pr_curves.items() \
                         if k != 'kwargs' and k != 'concatenated'])
            base_recall = np.linspace(0, 1, max_n)

            for k, v in pr_curves.items():
                if k == 'kwargs' or k == 'concatenated': continue
                recall = v['pr_curve'][0]
                precision = v['pr_curve'][1]
                # Interp precision so it fits the right shape for base_recall
                precision = np.interp(base_recall, recall, precision)
                precision[0] = 0
                # Saving to the list so we can stack and compute the mean and std
                precisions.append(precision)
                aucs.append(v['auc'])

    elif type(pr_curves) == list:
        # TODO FIX
        # THIS HERE ASSUMES THE RESULTS ARE IN FORMAT [((recall, precision), auc) ...]
        max_n = max([len(x[0][0]) for x in pr_curves])
        base_recall = np.linspace(0, 1, max_n)

        for curves in pr_curves:
            recall = curves[0][0]
            precision = curves[0][1]
            # Interp precision so it fits the right shape for base_recall
            precision = np.interp(base_recall, recall, precision)
            precision[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            precisions.append(precision)
            aucs.append(curves[1])

    mean_auc = np.mean(aucs)
    precisions = np.stack(precisions)
    mean_precisions = precisions.mean(axis=0)
    std_precisions = precisions.std(axis=0)
    upper = np.minimum(mean_precisions + std_precisions, 1)
    lower = mean_precisions - std_precisions
    return base_recall, mean_precisions, lower, upper, mean_auc


def get_roc(df, score='pred', target='agg_label', binder=None, anchor_mutation=None):
    """
    Args:
        df: DF containing the prediction or scores
        score: Name of the score columns, 'pred' by default
        target: Name of the target column, 'pred' by default
        binder: None, "Improved" or "Conserved" ; None by default
        anchor_mutation: None, True, False ; None by default

    Returns:

    """
    if binder is not None and anchor_mutation is not None:
        df = df.query('binder==@binder and anchor_mutation==@anchor_mutation').copy()
    try:
        fpr, tpr, _ = roc_curve(df[target].values, df[score].values)
        auc = roc_auc_score(df[target].values, df[score].values)
        auc01 = roc_auc_score(df[target].values, df[score].values, max_fpr=0.1)
    except KeyError:
        print('here')
        try:
            fpr, tpr, _ = roc_curve(df[target].values, df['mean_pred'].values)
            auc = roc_auc_score(df[target].values, df['mean_pred'].values)
            auc01 = roc_auc_score(df[target].values, df['mean_pred'].values, max_fpr=0.1)
        except:
            raise KeyError(f'{target} or "mean_pred" not in df\'s columns!')
    output = {"roc": (fpr, tpr),
              "auc": auc,
              "auc01": auc01,
              "npep": len(df)}
    return output
