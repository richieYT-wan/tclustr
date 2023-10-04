import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, V_MAP, J_MAP


class CDR3BetaDataset(Dataset):
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
            x = torch.cat([x,x_pep], dim=1)
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def get_dataset(self):
        return self

    def get_dataloader(self, batch_size, sampler):
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler(self))
        return dataloader


class PairedDataset(Dataset):
    """
    For now, only use CDR3b
    """

    def __init__(self, df, max_len_b=23, max_len_a=24, max_len_pep=12, encoding='BL50LO', pad_scale=None,
                 cdr3b_col='B3', cdr3a_col='A3', use_b=True, use_a=True, use_pep=True, use_v=False, use_j=False,
                 v_col='TRBV_gene', j_col='TRBJ_gene', v_dim=51, j_dim=13, v_map=V_MAP, j_map=J_MAP):
        super(PairedDataset, self).__init__()
        self.max_len_b = max_len_b
        self.max_len_a = max_len_a
        self.max_len_pep = max_len_pep
        self.use_b = use_b
        self.use_a = use_a
        self.use_pep = use_pep
        self.encoding = encoding
        self.pad_scale = pad_scale
        self.use_v = use_v
        self.use_j = use_j
        self.v_map = {k: v for v, k in enumerate(sorted(df[v_col].unique()))} if (v_map is None and use_v) else v_map
        self.j_map = {k: v for v, k in enumerate(sorted(df[j_col].unique()))} if (j_map is None and use_j) else j_map

        self.v_dim = v_dim
        self.j_dim = j_dim
        df['len_b'] = df[cdr3b_col].apply(len)
        df['len_a'] = df[cdr3a_col].apply(len)
        df = df.query('len_b<=@max_len_b and len_a<=@max_len_a')
        self.df = df
        # Only get sequences, no target because unsupervised learning, flattened to concat to classes
        x_b = encode_batch(df[cdr3b_col], max_len_b, encoding, pad_scale).flatten(start_dim=1) if use_b \
              else torch.empty([len(df), 0])
        x_a = encode_batch(df[cdr3a_col], max_len_a, encoding, pad_scale).flatten(start_dim=1) if use_a \
              else torch.empty([len(df), 0])
        x_pep = encode_batch(df['peptide'], max_len_pep, encoding, pad_scale).flatten(start_dim=1) if use_pep \
                else torch.empty([len(df), 0])
        x = torch.cat([x_b, x_a, x_pep], dim=1)
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def get_dataset(self):
        return self

    def get_dataloader(self, batch_size, sampler):
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler(self))
        return dataloader
