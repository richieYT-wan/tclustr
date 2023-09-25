import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, encode_batch_weighted, ics_dict, get_positional_encoding, pad_tensor, \
    get_ic_weights


class CDR3BetaDataset(Dataset):
    """
    For now, only use CDR3b
    """

    def __init__(self, df, max_len=23, encoding='BL50LO', pad_scale=None, cdr3b_col='B3', use_v=False, use_j=False,
                 v_col='TRBV_gene', j_col='TRBJ_gene', v_dim=51, j_dim=13):
        super(CDR3BetaDataset, self).__init__()
        self.max_len = max_len
        self.encoding = encoding
        self.pad_scale = pad_scale
        self.use_v = use_v
        self.use_j = use_j
        self.v_map = None
        self.j_map = None
        self.v_dim = v_dim
        self.j_dim = j_dim
        df['len'] = df[cdr3b_col].apply(len)
        df = df.query('len<=@max_len')
        self.df = df
        # Only get sequences, no target because unsupervised learning, flattened to concat to classes
        x = encode_batch(df[cdr3b_col], max_len, encoding, pad_scale).flatten(start_dim=1)
        if use_v:
            # get the mapping to a class
            self.v_map = {k: v for v, k in enumerate(sorted(df[v_col].unique()))}
            df['v_class'] = df[v_col].map(self.v_map).astype(int)
            x_v = F.one_hot(torch.from_numpy(df['v_class'].values), num_classes=v_dim).float()
            x = torch.cat([x, x_v], dim=1)
            self.x_v = x_v

        if use_j:
            # get the mapping to a class
            self.j_map = {k: v for v, k in enumerate(sorted(df[j_col].unique()))}
            df['j_class'] = df[j_col].map(self.j_map).astype(int)
            x_j = F.one_hot(torch.from_numpy(df['j_class'].values), num_classes=j_dim).float()
            x = torch.cat([x, x_j], dim=1)
            self.x_j = x_j

        # print(x[0], x[0].shape)
        # print('??\n')
        # print('\n',x[0][:460])
        # print('\n',x[0][460:460 + 51])
        # print('\n',x[0][460+51:])
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # TODO : Here, I return x twice because I'm too lazy to change everything in train_eval.py
        #        For the sake of memory and efficiency, this should be changed along with code in train_eval.py
        return self.x[idx], self.x[idx]

    def get_dataset(self):
        return self

    def get_dataloader(self, batch_size, sampler):
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler(self))
        return dataloader