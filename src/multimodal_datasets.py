from typing import Union

import numpy as np
import torch
from overrides import override

from src.data_processing import encode_batch, PEP_MAP2, batch_positional_encode, batch_encode_cat
from src.datasets import VAEDataset
from src.multimodal_models import BSSVAE, JMVAE


class TrimodalPepTCRDataset(VAEDataset):
    """
    Dataset that handles multi-modal data (Alpha chain, Beta chain, +/- peptide) ?
    For now, no peptide would probably mean health patient data
    Otherwise, could artificially increase data by removing randomly peptide labels
    """

    def __init__(self, df, max_len_a1=7, max_len_a2=8, max_len_a3=22,
                 max_len_b1=6, max_len_b2=7, max_len_b3=23, max_len_pep=12,
                 encoding='BL50LO', pad_scale=None, pep_encoding='BL50LO',
                 pep_pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                 pep_col='peptide', label_col='binder', pep_weighted=False, cat_method='pad_first'):
        super(TrimodalPepTCRDataset, self).__init__(df)
        assert not all([x == 0 for x in [max_len_a1, max_len_a2, max_len_a3,
                                         max_len_b1, max_len_b2, max_len_b3, max_len_pep]]), \
            'All loops max_len are 0! No chains will be added'
        assert cat_method in ['cat_first', 'pad_first'], "cat_method should be 'cat_first' or 'pad_first'"
        self.pad_scale = pad_scale
        self.max_len_a1 = max_len_a1
        self.max_len_a2 = max_len_a2
        self.max_len_a3 = max_len_a3
        self.max_len_b1 = max_len_b1
        self.max_len_b2 = max_len_b2
        self.max_len_b3 = max_len_b3
        self.max_len_pep = max_len_pep
        self.matrix_dim = 20
        self.aa_dim = 20
        # For now, disable positional encoding to make my life easier seeing how we have 3 separate encoders
        # and thus shouldn't need to explicitely tell a model which chain is what
        # self.add_positional_encoding = add_positional_encoding
        mldict = {k: v for k, v in
                  zip([a1_col, a2_col, a3_col, b1_col, b2_col, b3_col, pep_col],
                      [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])}
        acols = [x for x in mldict if x.startswith('A') and mldict[x] > 0]
        bcols = [x for x in mldict if x.startswith('B') and mldict[x] > 0]
        # Cat first or pad first?
        if cat_method == 'pad_first':
            # This max length dict with colname:max_len is used to iterate
            # but also to set the padding vector in pos encoding

            # Filter DF based on seq max lengths
            for seq_col, max_len, in mldict.items():
                if max_len > 0:
                    df['len_q'] = df[seq_col].apply(len)
                    df = df.query('len_q <= @max_len')
            self.df = df.drop(columns=['len_q']).reset_index(drop=True)
            # Transform this to dictionary instead
            x_seq = {}

            # Building the sequence tensor and (if applicable) positional tensor as 2D tensor
            # Then at the very end, stack, cat, flatten as needed
            for seq_col, max_len in mldict.items():
                if max_len > 0:
                    seq_tensor = encode_batch(df[seq_col].values, max_len, encoding, pad_scale)
                    x_seq[seq_col] = seq_tensor

            # Concatenate all the tensors in the list `x_seq` into one tensor `x_seq`
            # Need to concatenate each sub-sequences (A, B, pep) together
            x_alpha = torch.cat([x_seq[k] for k in acols], dim=1)
            x_beta = torch.cat([x_seq[k] for k in bcols], dim=1)
            x_pep = x_seq[pep_col]
        # do not use this for now (cat_first) !
        elif cat_method == 'cat_first':
            df['alpha_sequence'] = df[acols].sum(axis=1)
            df['beta_sequence'] = df[bcols].sum(axis=1)
            x_alpha = encode_batch(df['alpha_sequence'], sum([v for k, v in mldict if k in acols]),
                                   encoding, pad_scale)
            x_beta = encode_batch(df['beta_sequence'], sum([v for k, v in mldict if k in bcols]),
                                  encoding, pad_scale)
            x_pep = encode_batch(df[pep_col], max_len_pep,
                                 encoding, pad_scale)
        else:
            raise ValueError('No proper cat method provided or all seqs empty')

        self.x_alpha = x_alpha.flatten(start_dim=1)
        self.x_beta = x_beta.flatten(start_dim=1)
        self.x_pep = x_pep.flatten(start_dim=1)
        # Create flags to filter modalities ; Don't use "input_type" from Mathias' dataset to make it more flexible

        self.mask_alpha = torch.from_numpy(df[acols].sum(axis=1).apply(lambda x: not all([z == "X" for z in x])).values)
        self.mask_beta = torch.from_numpy(df[bcols].sum(axis=1).apply(lambda x: not all([z == "X" for z in x])).values)
        self.mask_pep = torch.from_numpy(df[pep_col].apply(lambda x: not all([z == "X" for z in x])).values)
        self.df = df
        self.len = len(df)

        self.pep_weighted = pep_weighted
        self.labels = torch.from_numpy(self.df[pep_col].map(PEP_MAP2).values)
        self.binder = torch.from_numpy(self.df[label_col].values).unsqueeze(1).float()
        # TODO: fix this
        #   Quick & Dirty fix for triplet loss : Use PepWeights as binary mask to remove some losses
        #   Set pepweights as where original_pep == pep, in TwoStageVAELoss, use weights only for triplet
        #   So that we don't learn triplet loss for swapped negatives
        self.pep_weights = torch.from_numpy(
            self.df.apply(lambda x: x['original_peptide'] == x['peptide'], axis=1).values).float().unsqueeze(1)

    @override
    def __getitem__(self, idx):
        """
        Args:
            idx:

        Returns:
            tensors and masks
        """
        # TODO: Currently, the return order for pep_weighted is not unified across the various classes / types
        #       Sometimes, it's returned last (in TwoStage and Trimodal), sometimes second to last (Triplet / normal)
        if self.pep_weighted:
            return self.x_alpha[idx], self.x_beta[idx], self.x_pep[idx], \
                   self.mask_alpha[idx], self.mask_beta[idx], self.mask_pep[idx], \
                   self.labels[idx], self.pep_weights[idx]
        else:
            return self.x_alpha[idx], self.x_beta[idx], self.x_pep[idx], \
                   self.mask_alpha[idx], self.mask_beta[idx], self.mask_pep[idx], \
                   self.labels[idx]


class MultimodalPepTCRDataset(VAEDataset):
    # TODO : Add paired only warm-up
    #        use self.counter() to return only paired for X epochs, then
    #        make it increasingly difficult by returning unpaired?
    #        Remains the problem for KLD on a per-datapoint basis --> incr batchsize?

    def __init__(self, df, max_len_a1=7, max_len_a2=8, max_len_a3=22,
                 max_len_b1=6, max_len_b2=7, max_len_b3=23, max_len_pep=12,
                 encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2',
                 b3_col='B3', pair_only=False, return_pair=False,
                 pep_col='peptide', add_positional_encoding=False):
        super(MultimodalPepTCRDataset, self).__init__(df)
        assert not all([x == 0 for x in [max_len_a1, max_len_a2, max_len_a3,
                                         max_len_b1, max_len_b2, max_len_b3, max_len_pep]]), \
            'All loops max_len are 0! No chains will be added'

        self.pair_only = pair_only
        self.return_pair = return_pair

        if pair_only:
            df_tcr_only = df.query('input_type=="tcr_pep"')
            df_pep_only = df.query('input_type=="tcr_pep"')
            df_pep_tcr = df.query('input_type=="tcr_pep"')
        else:
            df_tcr_only = df.query('input_type=="tcr"')
            df_pep_only = df.query('input_type=="pep"')
            df_pep_tcr = df.query('input_type=="tcr_pep"')

        self.pad_scale = pad_scale
        self.max_len_a1 = max_len_a1
        self.max_len_a2 = max_len_a2
        self.max_len_a3 = max_len_a3
        self.max_len_b1 = max_len_b1
        self.max_len_b2 = max_len_b2
        self.max_len_b3 = max_len_b3
        self.max_len_pep = max_len_pep
        self.matrix_dim = 20
        self.aa_dim = 20
        tcr_len = sum([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3])
        self.max_len_tcr = tcr_len
        # Max len pep == 0 to encode only TCR
        df_tcr_only, x_tcr_marg, x_tcr_matrix_dim = _encode_chains(df_tcr_only, encoding, pad_scale,
                                                                   a1_col, a2_col, a3_col, b1_col, b2_col, b3_col,
                                                                   pep_col,
                                                                   max_len_a1, max_len_a2, max_len_a3,
                                                                   max_len_b1, max_len_b2, max_len_b3,
                                                                   max_len_pep=0,
                                                                   add_positional_encoding=add_positional_encoding)
        # max_len chains == 0 to encode only pep
        df_pep_only, x_pep_marg, x_pep_matrix_dim = _encode_chains(df_pep_only, encoding, pad_scale,
                                                                   a1_col, a2_col, a3_col, b1_col, b2_col, b3_col,
                                                                   pep_col,
                                                                   max_len_a1=0, max_len_a2=0, max_len_a3=0,
                                                                   max_len_b1=0, max_len_b2=0, max_len_b3=0,
                                                                   max_len_pep=max_len_pep,
                                                                   add_positional_encoding=add_positional_encoding)
        # Set all chains to their ml and encode and slice
        df_pep_tcr, x_tcr_pep_joint, x_tcrpep_matrix_dim = _encode_chains(df_pep_tcr, encoding, pad_scale,
                                                                          a1_col, a2_col, a3_col, b1_col, b2_col,
                                                                          b3_col,
                                                                          pep_col,
                                                                          max_len_a1, max_len_a2, max_len_a3,
                                                                          max_len_b1, max_len_b2, max_len_b3,
                                                                          max_len_pep, add_positional_encoding)
        x_tcr_joint = x_tcr_pep_joint[:, :tcr_len, :]
        x_pep_joint = x_tcr_pep_joint[:, tcr_len:, :]
        # Save each modality into a tensor and use sub-sampling
        self.x_tcr_marg = x_tcr_marg
        self.x_pep_marg = x_pep_marg
        self.x_tcr_joint = x_tcr_joint
        self.x_pep_joint = x_pep_joint
        # Saving the various number of sample and dfs for various purposes
        self.n_paired = len(df_pep_tcr)
        self.n_tcr_marg = len(df_tcr_only)
        self.n_pep_marg = len(df_pep_only)
        self.df_tcr_only = df_tcr_only
        self.df_pep_only = df_pep_only
        self.df_pep_tcr = df_pep_tcr
        # initializing some random sub-sampled index
        self.tcr_indices = _randindex(self.n_tcr_marg, self.n_paired, random_state=0)
        self.pep_indices = _randindex(self.n_pep_marg, self.n_paired, random_state=0)

        # self.pep_weights = torch.from_numpy(
        #     self.df.apply(lambda x: x['original_peptide'] == x['peptide'], axis=1).values).float().unsqueeze(1)

    @override
    def __getitem__(self, idx):
        # TODO : Do "predict" mode where we return ALL instances for the marginal in a way that makes sense
        """
        Uses renewed self.tcr/pep_indices at each epoch to do the subsampling part to match self.n_paired
        Args:
            idx:
        Returns:
            tensors:  x_tcr_marg, x_tcr_joint, x_pep_joint, x_pep_marg (follows the order of graph left to right)
        """
        if self.pair_only:
            if self.return_pair:
                return self.x_tcr_joint[idx], self.x_pep_joint[idx]
            else:
                return self.x_tcr_joint[idx], self.x_tcr_joint[idx], self.x_pep_joint[idx], self.x_pep_joint[idx]
        else:
            return self.x_tcr_marg[self.tcr_indices[idx]], \
                   self.x_tcr_joint[idx], \
                   self.x_pep_joint[idx], \
                   self.x_pep_marg[self.pep_indices[idx]]

    @override
    def __len__(self):
        return self.n_paired

    @override
    def increment_counter(self):
        self.counter += 1
        # update random sample, using current counter as random_state to keep the sampling reproducible
        self.tcr_indices = _randindex(self.n_tcr_marg, self.n_paired, random_state=self.counter)
        self.pep_indices = _randindex(self.n_pep_marg, self.n_paired, random_state=self.counter)


class MultimodalCLFLatentDataset(MultimodalPepTCRDataset):
    # weight = 5.33 for the 400 datapoints dataset, tested using
    # loss_tensor = []
    # for divider in torch.linspace(5.3,5.4, 50):
    #     pepweights = (np.log2(len(df) / df.groupby(['peptide']).agg(count=(f'B3', 'count'))) / divider.item() ).round(4).to_dict()['count']
    #     for k in pepweights:
    #         loss_tensor.extend([pepweights[k]]*count[k])
    #     print(divider, torch.tensor(loss_tensor).mean())

    def __init__(self, model: Union[BSSVAE, JMVAE], df, max_len_a1=7, max_len_a2=8, max_len_a3=22,
                 max_len_b1=6, max_len_b2=7, max_len_b3=23, max_len_pep=12,
                 encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2',
                 b3_col='B3', pep_encoding=False, pep_weighted=False, pep_weight_scale=5.33,
                 pep_col='peptide', add_positional_encoding=False):
        super(MultimodalCLFLatentDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3,
                                                         max_len_b1, max_len_b2, max_len_b3, max_len_pep,
                                                         encoding, pad_scale,
                                                         a1_col, a2_col, a3_col, b1_col, b2_col, b3_col,
                                                         pair_only=True, return_pair=False, pep_col=pep_col,
                                                         add_positional_encoding=add_positional_encoding)
        with torch.no_grad():
            model.eval()
            self.z = model.embed(self.x_tcr_joint, self.x_pep_joint, 'joint').detach()
            if pep_encoding == 'none':
                encoded_peps = torch.empty([len(self.z), 0])
            elif pep_encoding == 'categorical':
                # dim (N, 12)
                encoded_peps = batch_encode_cat(df[pep_col], 12, -1)
            else:
                # dim (N, 12, 20) -> (N, 240) after flatten
                encoded_peps = encode_batch(df[pep_col], 12, pep_encoding, -20).flatten(start_dim=1)
            self.z = torch.cat([self.z, encoded_peps], dim = 1)
        self.labels = torch.from_numpy(self.df_pep_tcr['binder'].values).unsqueeze(1).float()
        print(f'PEPWEIGHTED {pep_weighted}')
        self.pep_weighted = pep_weighted
        if pep_weighted:
            pepweights = (np.log2(len(self.df_pep_tcr) / self.df_pep_tcr.groupby(['peptide']) \
                                                             .agg(count=(f'{b3_col}', 'count'))) / pep_weight_scale) \
                                                .round(5).to_dict()['count']
            print('HERE PEPWEIGHTED')
            self.pep_weights = torch.from_numpy(self.df_pep_tcr['peptide'].map(pepweights).values)

        del self.x_tcr_joint, self.x_tcr_marg, self.x_pep_joint, self.x_pep_marg, self.df_tcr_only, self.df_pep_only

    def __getitem__(self, idx):
        if self.pep_weighted:
            return self.z[idx], self.pep_weights[idx], self.labels[idx]
        else:
            return self.z[idx], self.labels[idx]

    def __len__(self):
        return len(self.z)


def _encode_chains(df, encoding, pad_scale, a1_col, a2_col, a3_col, b1_col, b2_col, b3_col, pep_col,
                   max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep,
                   add_positional_encoding=False):
    mldict = {k: v for k, v in
              zip([a1_col, a2_col, a3_col, b1_col, b2_col, b3_col, pep_col],
                  [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])}
    # Filter DF based on seq max lengths
    for seq_col, max_len, in mldict.items():
        if max_len > 0:
            df['len_q'] = df[seq_col].apply(len)
            df = df.query('len_q <= @max_len')
    df = df.drop(columns=['len_q']).reset_index(drop=True)
    x_seq = []
    x_pos = []
    # I put this shit here because PyCharm is being an asshole with the if scope + global statement squiggly fucking line
    pad_values = {}
    if add_positional_encoding:
        # Selected columns are those whose maxlen>0 (i.e. are used)
        selected_columns = [k for k, v in mldict.items() if v > 0]
        # Pre-setting the left-right pad tuples depending on which columns are used
        max_lens_values = [mldict[k] for k in selected_columns]
        pad_values = {k: (sum(max_lens_values[:i]), sum(max_lens_values) - sum(max_lens_values[:i])) \
                      for i, k in enumerate(selected_columns)}

    # Building the sequence tensor and (if applicable) positional tensor as 2D tensor
    # Then at the very end, stack, cat, flatten as needed
    for seq_col, max_len in mldict.items():
        if max_len > 0:
            seq_tensor = encode_batch(df[seq_col].values, max_len, encoding, pad_scale)
            x_seq.append(seq_tensor)
            if add_positional_encoding:
                pos_tensor = batch_positional_encode(df[seq_col].values, pad_values[seq_col])
                x_pos.append(pos_tensor)

    # Concatenate all the tensors in the list `x_seq` into one tensor `x_seq`
    x_seq = torch.cat(x_seq, dim=1)
    matrix_dim = 20
    if add_positional_encoding:
        # Stack the `x_pos` tensors in the list together into a single tensor along dimension 2 (n_chains)
        matrix_dim += len(x_pos)
        x_pos = torch.stack(x_pos, dim=2)
        # Add the pos encode to the seq tensor (N, sum(ML), 20) -> (N, sum(ML), 20+n_chains)
        x_seq = torch.cat([x_seq, x_pos], dim=2)

    return df, x_seq, matrix_dim


def _randindex(data_size, n_samples, random_state=None):
    """
    Randomizes a set of indices to sample each epoch for the multimodal case where we subsample the other modalities
    Args:
        data_size:
        n_samples:

    Returns:
        indices (torch.tensor)
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    return torch.randperm(data_size)[:n_samples]
