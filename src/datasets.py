import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, batch_encode_cat, V_MAP, J_MAP, PEP_MAP, encoding_matrix_dict
from overrides import override


class VAEDataset(Dataset):
    """
    Parent class so I don't have to re-declare the same bound methods
    """

    def __init__(self, x=torch.empty([10, 1])):
        super(VAEDataset, self).__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def get_dataset(self):
        return self

    def get_dataloader(self, batch_size, sampler, **kwargs):
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler(self), **kwargs)
        return dataloader


class CDR3BetaDataset(VAEDataset):
    """
    For now, only use CDR3b
    """

    def __init__(self, df, max_len=23, encoding='BL50LO', pad_scale=None, cdr3b_col='B3', use_v=True, use_j=True,
                 v_col='TRBV_gene', j_col='TRBJ_gene', v_dim=51, j_dim=13, v_map=V_MAP, j_map=J_MAP, add_pep=False,
                 max_len_pep=12):
        super(CDR3BetaDataset, self).__init__()
        self.max_len = max_len
        self.encoding = encoding

        self.pad_scale = pad_scale
        self.use_v = use_v
        self.use_j = use_j
        self.v_map = {k: v for v, k in enumerate(sorted(df[v_col].unique()))} if (v_map is None and use_v) else v_map
        self.j_map = {k: v for v, k in enumerate(sorted(df[j_col].unique()))} if (j_map is None and use_j) else j_map

        self.v_dim = v_dim
        self.j_dim = j_dim
        self.df = df
        # Only get sequences, no target because unsupervised learning, flattened to concat to classes
        df['len'] = df[cdr3b_col].apply(len)
        df = df.query('len<=@max_len')
        x = encode_batch(df[cdr3b_col], max_len, encoding, pad_scale).flatten(start_dim=1)
        if add_pep:
            x_pep = encode_batch(df['peptide'], max_len_pep, encoding, pad_scale).flatten(start_dim=1)
            x = torch.cat([x, x_pep], dim=1)
        if use_v:
            # get the mapping to a class
            df['v_class'] = df[v_col].map(self.v_map).astype(int)
            x_v = F.one_hot(torch.from_numpy(df['v_class'].values), num_classes=v_dim).float()
            x = torch.cat([x, x_v], dim=1)
            self.x_v = x_v

        if use_j:
            # get the mapping to a class
            df['j_class'] = df[j_col].map(self.j_map).astype(int)
            x_j = F.one_hot(torch.from_numpy(df['j_class'].values), num_classes=j_dim).float()
            x = torch.cat([x, x_j], dim=1)
            self.x_j = x_j

        self.x = x


class FullTCRDataset(VAEDataset):

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, encoding='BL50LO',
                 pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                 pep_weighted=False, pep_weight_scale=3.8):
        super(FullTCRDataset, self).__init__()
        # TODO : Current behaviour If max_len_x = 0, then don't use that chain...
        #        Is that the most elegant way to do this ?
        assert not all([x == 0 for x in [max_len_a1, max_len_a2, max_len_a3,
                                         max_len_b1, max_len_b2, max_len_b3]]), \
            'All loops max_len are 0! No chains will be added'

        x_seq = []
        self.max_len_a1 = max_len_a1
        self.max_len_a2 = max_len_a2
        self.max_len_a3 = max_len_a3
        self.max_len_b1 = max_len_b1
        self.max_len_b2 = max_len_b2
        self.max_len_b3 = max_len_b3
        self.use_a1 = not (max_len_a1 == 0)
        self.use_a2 = not (max_len_a2 == 0)
        self.use_a3 = not (max_len_a3 == 0)
        self.use_b1 = not (max_len_b1 == 0)
        self.use_b2 = not (max_len_b2 == 0)
        self.use_b3 = not (max_len_b3 == 0)

        # bad double loop because brain slow
        for max_len, seq_col in zip([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3],
                                    [a1_col, a2_col, a3_col, b1_col, b2_col, b3_col]):
            if max_len != 0:
                df['len_q'] = df[seq_col].apply(len)
                df = df.query('len_q <= @max_len')
        for max_len, seq_col in zip([max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3],
                                    [a1_col, a2_col, a3_col, b1_col, b2_col, b3_col]):
            if max_len != 0:
                x_seq.append(encode_batch(df[seq_col], max_len, encoding, pad_scale).flatten(start_dim=1))

        self.df = df.drop(columns=['len_q']).reset_index(drop=True)
        self.x = torch.cat(x_seq, dim=1)

        # Here save a weight that is peptide specific to give more/less importance to peptides that are less/more frequent
        if pep_weighted:
            pepweights = np.log2(len(self.df) / self.df.groupby(['peptide']).agg(count=(f'{b3_col}', 'count')))
            self.pep_weights = torch.from_numpy(
                self.df.apply(lambda x: pepweights.loc[x['peptide']], axis=1).values / pep_weight_scale).flatten(
                start_dim=0)
        else:
            self.pep_weights = torch.empty([len(self.x)])


class TCRSpecificDataset(FullTCRDataset):
    """
    This class should be used for Triplet-loss optimization as well as optimal VAE-MLP models
    """

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, encoding='BL50LO',
                 pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                 pep_weighted=False, pep_weight_scale=3.8):
        super(TCRSpecificDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                 max_len_b3, encoding, pad_scale, a1_col, a2_col, a3_col, b1_col,
                                                 b2_col, b3_col, pep_weighted, pep_weight_scale)
        # Here "labels" are for each peptide, used for the triplet loss
        self.labels = torch.from_numpy(df['peptide'].map(PEP_MAP).values)

    @override
    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


class BimodalTCRpMHCDataset(TCRSpecificDataset):

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, encoding='BL50LO',
                 pad_scale=None, pep_encoding='categorical', pep_pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3',
                 b1_col='B1', b2_col='B2', b3_col='B3', pep_col='peptide', label_col='binder', pep_weighted=False,
                 pep_weight_scale=3.8):
        super(BimodalTCRpMHCDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                    max_len_b3, encoding, pad_scale, a1_col, a2_col, a3_col, b1_col,
                                                    b2_col, b3_col, pep_weighted, pep_weight_scale)
        assert pep_encoding in ['categorical'] + list(
            encoding_matrix_dict.keys()), f'Encoding for peptide {pep_encoding} not recognized.' \
                                          f"Must be one of {['categorical'] + list(encoding_matrix_dict.keys())}"
        if pep_encoding == 'categorical':
            encoded_peps = batch_encode_cat(df[pep_col], 12, -1)
        else:
            encoded_peps = encode_batch(df[pep_col], 12, pep_encoding, pep_pad_scale)
        # Inherits self.x and self.label from its parent class)
        self.labels = torch.from_numpy(df['original_peptide'].map(PEP_MAP).values)
        self.x_pep = encoded_peps
        self.binder = torch.from_numpy(df[label_col].values).unsqueeze(1).float()

    @override
    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            x : Encoded TCR sequence tensor (VAE)
            x_pep : encoded peptide sequence tensor (MLP)
            labels : Peptide label for triplet loss (VAE)
            binder : pMHC-TCR binary binder label for classification loss (MLP)
        """
        return self.x[idx], self.x_pep[idx], self.labels[idx], self.binder[idx]


class LatentTCRpMHCDataset(FullTCRDataset):
    """
    Placeholder where for now, we use a frozen VAE model so that the latent Z is always fixed
    """

    def __init__(self, model, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3,
                 encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2',
                 b3_col='B3', pep_col='peptide', label_col='binder', pep_encoding='categorical', pep_weighted=False,
                 pep_weight_scale=3.8):
        super(LatentTCRpMHCDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                   max_len_b3, encoding, pad_scale, a1_col, a2_col, a3_col, b1_col,
                                                   b2_col, b3_col, pep_weighted, pep_weight_scale)

        with torch.no_grad():
            model.eval()
            z_latent = model.embed(self.x)

        assert pep_encoding in ['categorical'] + list(
            encoding_matrix_dict.keys()), f'Encoding for peptide {pep_encoding} not recognized.' \
                                          f"Must be one of {['categorical'] + list(encoding_matrix_dict.keys())}"
        if pep_encoding == 'categorical':
            # dim (N, 12)
            encoded_peps = batch_encode_cat(df[pep_col], 12, -1)
        else:
            # dim (N, 12, 20) -> (N, 240) after flatten
            encoded_peps = encode_batch(df[pep_col], 12, pep_encoding, -20).flatten(start_dim=1)

        self.x = torch.cat([z_latent, encoded_peps], dim=1)
        # Here labels are "binders"
        self.labels = torch.from_numpy(df[label_col].values).unsqueeze(1).float()

    @override
    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]

#
# class PairedDataset(VAEDataset):
#     """
#     For now, only use CDR3b
#     """
#
#     def __init__(self, df, max_len_b=23, max_len_a=24, max_len_pep=12, encoding='BL50LO', pad_scale=None,
#                  cdr3b_col='TRB_CDR3', cdr3a_col='TRA_CDR3', pep_col='peptide', use_b=True, use_a=True, use_pep=True,
#                  use_v=False, use_j=False,
#                  v_col='TRBV_gene', j_col='TRBJ_gene', v_dim=51, j_dim=13, v_map=V_MAP, j_map=J_MAP):
#         super(PairedDataset, self).__init__()
#         self.max_len_b = max_len_b
#         self.max_len_a = max_len_a
#         self.max_len_pep = max_len_pep
#         self.use_b = use_b
#         self.use_a = use_a
#         self.use_pep = use_pep
#         self.encoding = encoding
#         self.pad_scale = pad_scale
#         self.use_v = use_v
#         self.use_j = use_j
#         self.v_map = {k: v for v, k in enumerate(sorted(df[v_col].unique()))} if (v_map is None and use_v) else v_map
#         self.j_map = {k: v for v, k in enumerate(sorted(df[j_col].unique()))} if (j_map is None and use_j) else j_map
#
#         self.v_dim = v_dim
#         self.j_dim = j_dim
#         df['len_b'] = df[cdr3b_col].apply(len) if use_b else 0
#         df['len_a'] = df[cdr3a_col].apply(len) if use_a else 0
#         df['len_pep'] = df[pep_col].apply(len) if use_pep else 0
#         df = df.query('len_b<=@max_len_b and len_a<=@max_len_a and len_pep<=@max_len_pep')
#         self.df = df
#         # Only get sequences, no target because unsupervised learning, flattened to concat to classes
#         x_b = encode_batch(df[cdr3b_col], max_len_b, encoding, pad_scale).flatten(start_dim=1) if use_b \
#             else torch.empty([len(df), 0])
#         x_a = encode_batch(df[cdr3a_col], max_len_a, encoding, pad_scale).flatten(start_dim=1) if use_a \
#             else torch.empty([len(df), 0])
#         x_pep = encode_batch(df[pep_col], max_len_pep, encoding, pad_scale).flatten(start_dim=1) if use_pep \
#             else torch.empty([len(df), 0])
#         x = torch.cat([x_b, x_a, x_pep], dim=1)
#         if use_v:
#             # get the mapping to a class
#             df['v_class'] = df[v_col].map(self.v_map).astype(int)
#             x_v = F.one_hot(torch.from_numpy(df['v_class'].values), num_classes=v_dim).float()
#             x = torch.cat([x, x_v], dim=1)
#             self.x_v = x_v
#
#         if use_j:
#             # get the mapping to a class
#             df['j_class'] = df[j_col].map(self.j_map).astype(int)
#             x_j = F.one_hot(torch.from_numpy(df['j_class'].values), num_classes=j_dim).float()
#             x = torch.cat([x, x_j], dim=1)
#             self.x_j = x_j
#
#         self.x = x
