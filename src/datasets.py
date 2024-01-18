import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, batch_encode_cat, batch_positional_encode, V_MAP, J_MAP, PEP_MAP, \
    encoding_matrix_dict
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
    # TODO : Deprecated
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

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep=0,
                 add_positional_encoding=False, encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2',
                 a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3', pep_col='original_peptide', pep_weighted=False,
                 pep_weight_scale=3.8):
        super(FullTCRDataset, self).__init__()
        # TODO : Current behaviour If max_len_x = 0, then don't use that chain...
        #        Is that the most elegant way to do this ?
        assert not all([x == 0 for x in [max_len_a1, max_len_a2, max_len_a3,
                                         max_len_b1, max_len_b2, max_len_b3]]), \
            'All loops max_len are 0! No chains will be added'

        self.max_len_a1 = max_len_a1
        self.max_len_a2 = max_len_a2
        self.max_len_a3 = max_len_a3
        self.max_len_b1 = max_len_b1
        self.max_len_b2 = max_len_b2
        self.max_len_b3 = max_len_b3
        self.max_len_pep = max_len_pep
        self.add_positional_encoding = add_positional_encoding

        # This max length dict with colname:max_len is used to iterate but also to set the padding vector in pos encoding
        mldict = {k: v for k, v in
                  zip([a1_col, a2_col, a3_col, b1_col, b2_col, b3_col, pep_col],
                      [max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep])}
        # Filter DF based on seq max lengths
        for seq_col, max_len, in mldict.items():
            if max_len > 0:
                df['len_q'] = df[seq_col].apply(len)
                df = df.query('len_q <= @max_len')
        self.df = df.drop(columns=['len_q']).reset_index(drop=True)
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
        if add_positional_encoding:
            # Stack the `x_pos` tensors in the list together into a single tensor along dimension 2 (n_chains)
            x_pos = torch.stack(x_pos, dim=2)
            # Add the pos encode to the seq tensor (N, sum(ML), 20) -> (N, sum(ML), 20+n_chains)
            x_seq = torch.cat([x_seq, x_pos], dim=2)

        self.x = x_seq.flatten(start_dim=1)
        self.pep_weighted = pep_weighted
        # Here save a weight that is peptide specific to give more/less importance to peptides that are less/more frequent
        if pep_weighted:
            pepweights = np.log2(len(self.df) / self.df.groupby(['peptide']).agg(count=(f'{b3_col}', 'count')))
            self.pep_weights = torch.from_numpy(
                self.df.apply(lambda x: pepweights.loc[x['peptide']], axis=1).values / pep_weight_scale).flatten(
                start_dim=0)
        else:
            self.pep_weights = torch.ones([len(self.x)])


class TCRSpecificDataset(FullTCRDataset):
    """
    This class should be used for Triplet-loss optimization as well as optimal VAE-MLP models
    """

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep=0,
                 add_positional_encoding=False, encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2',
                 a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3', pep_weighted=False, pep_weight_scale=3.8):
        super(TCRSpecificDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                 max_len_b3, max_len_pep=max_len_pep,
                                                 add_positional_encoding=add_positional_encoding, encoding=encoding,
                                                 pad_scale=pad_scale, a1_col=a1_col, a2_col=a2_col, a3_col=a3_col,
                                                 b1_col=b1_col, b2_col=b2_col, b3_col=b3_col, pep_weighted=pep_weighted,
                                                 pep_weight_scale=pep_weight_scale)
        # Here "labels" are for each peptide, used for the triplet loss
        self.labels = torch.from_numpy(df['peptide'].map(PEP_MAP).values)

    @override
    def __getitem__(self, idx):
        if self.pep_weighted:
            return self.x[idx], self.pep_weights[idx], self.labels[idx]
        else:
            return self.x[idx], self.labels[idx]


class BimodalTCRpMHCDataset(TCRSpecificDataset):

    def __init__(self, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep=0,
                 add_positional_encoding=False, encoding='BL50LO', pad_scale=None, pep_encoding='BL50LO',
                 pep_pad_scale=None, a1_col='A1', a2_col='A2', a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3',
                 pep_col='peptide', label_col='binder', pep_weighted=False, pep_weight_scale=3.8):
        super(BimodalTCRpMHCDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                    max_len_b3, max_len_pep=max_len_pep,
                                                    add_positional_encoding=add_positional_encoding, encoding=encoding,
                                                    pad_scale=pad_scale, a1_col=a1_col, a2_col=a2_col, a3_col=a3_col,
                                                    b1_col=b1_col, b2_col=b2_col, b3_col=b3_col,
                                                    pep_weighted=pep_weighted, pep_weight_scale=pep_weight_scale)
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
        if self.pep_weighted:
            return self.x[idx], self.x_pep[idx], self.labels[idx], self.binder[idx], self.pep_weights[idx]
        else:
            return self.x[idx], self.x_pep[idx], self.labels[idx], self.binder[idx]


class LatentTCRpMHCDataset(FullTCRDataset):
    """
    Placeholder where for now, we use a frozen VAE model so that the latent Z is always fixed
    """

    def __init__(self, model, df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2, max_len_b3, max_len_pep=0,
                 add_positional_encoding=False, encoding='BL50LO', pad_scale=None, a1_col='A1', a2_col='A2',
                 a3_col='A3', b1_col='B1', b2_col='B2', b3_col='B3', pep_col='peptide', label_col='binder',
                 pep_encoding='BL50LO', pep_weighted=False, pep_weight_scale=3.8):
        super(LatentTCRpMHCDataset, self).__init__(df, max_len_a1, max_len_a2, max_len_a3, max_len_b1, max_len_b2,
                                                   max_len_b3, max_len_pep=max_len_pep,
                                                   add_positional_encoding=add_positional_encoding, encoding=encoding,
                                                   pad_scale=pad_scale, a1_col=a1_col, a2_col=a2_col, a3_col=a3_col,
                                                   b1_col=b1_col, b2_col=b2_col, b3_col=b3_col,
                                                   pep_weighted=pep_weighted, pep_weight_scale=pep_weight_scale)

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
        # Here labels are binary "binders" label, instead of pep_label for triplet loss
        self.labels = torch.from_numpy(df[label_col].values).unsqueeze(1).float()

    @override
    def __getitem__(self, idx):
        if self.pep_weighted:
            return self.x[idx], self.pep_weights[idx], self.labels[idx]
        else:
            return self.x[idx], self.labels[idx]
