import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, encode_batch_weighted, ics_dict, get_positional_encoding, pad_tensor, get_ic_weights


class MutWtDataset(Dataset):
    """
    Here for now, only get encoding and try to
    """

    def __init__(self, df: pd.DataFrame, max_len: int, encoding: str = 'onehot', mut_col: str = 'mutant',
                 wt_col: str = 'wild_type', target_col: str = 'target', pad_scale: float = None,
                 positional_encoding=False,
                 ic_weighted=False, feature_cols: list = None):
        super(MutWtDataset, self).__init__()
        # Encoding stuff
        if feature_cols is None:
            feature_cols = []
        df['len'] = df[mut_col].apply(len)
        df = df.query('len<=@max_len')
        # TODO: Implement the IC weights stuff at some point here
        #       to scale inputs for NNAlign with encode_batch_weighted()
        if ic_weighted:
            x_mut = encode_batch_weighted(df, ics_dict=ics_dict['KL'], max_len=max_len, encoding=encoding,
                                          pad_scale=pad_scale,
                                          seq_col=mut_col, hla_col='HLA', target_col=target_col, mask=True,
                                          invert=False,
                                          threshold=0.2)
            x_wt = encode_batch_weighted(df, ics_dict=ics_dict['KL'], max_len=max_len, encoding=encoding,
                                         pad_scale=pad_scale,
                                         seq_col=wt_col, hla_col='HLA', target_col=target_col, mask=True, invert=False,
                                         threshold=0.2)
        else:
            x_mut = encode_batch(df[mut_col], max_len, encoding, pad_scale)
            x_wt = encode_batch(df[wt_col], max_len, encoding, pad_scale)

        self.positional_encoding = positional_encoding
        self.pe_mut = get_positional_encoding(x_mut, pad_scale) if positional_encoding else None
        self.pe_wt = get_positional_encoding(x_wt, pad_scale) if positional_encoding else None

        # Mask for padding ; Allows to standardize the sequence features without considering the pads
        # TODO: Is this behaviour actually correct?
        # Creating the mask to allow selection of kmers without padding
        # Here, using a single mask because in theory WT and MUT are aligned and have the same length thus same pad
        x_mask = torch.from_numpy(df['len'].values) - 1
        range_tensor = torch.arange(max_len).unsqueeze(0).repeat(len(x_mut), 1)
        self.x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)

        self.x_mut = x_mut.contiguous()
        self.x_wt = x_wt.contiguous()
        self.y = y.contiguous()
        #
        # Add extra features
        if len(feature_cols) > 0:
            self.x_features = torch.from_numpy(df[feature_cols].values).float()
            self.extra_features_flag = True
        else:
            self.x_features = torch.empty((len(x_mut),))
            self.extra_features_flag = False

        # Saving df in case it's needed
        self.df = df
        self.len = len(y)
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Returns the appropriate input tensors (X_mut, x_wt, y)
        :param idx:
        :return:
        """

        if self.extra_features_flag:
            pass
        else:
            return self.x_mut[idx], self.x_wt[idx], self.y[idx]

    def get_std_tensors(self):
        if self.extra_features_flag:
            pass
        else:
            return self.x_mut, self.x_wt, self.x_mask


def get_mutwt_dataloader(df: pd.DataFrame, max_len: int, encoding: str = 'onehot', mut_col: str = 'mutant',
                         wt_col: str = 'wild_type', target_col: str = 'target', pad_scale: float = None,
                         ic_weighted=False, feature_cols: list = None, batch_size=64,
                         sampler=torch.utils.data.RandomSampler, return_dataset=False):
    dataset = MutWtDataset(df, max_len, encoding, mut_col, wt_col, target_col, pad_scale,
                           ic_weighted=ic_weighted, feature_cols=feature_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader


# ESM stuff

class ESMDataset(Dataset):
    """
    Class for ESM dataset using the pre-computed ESM embeddings
    """

    def __init__(self, df, datapath='../data/esm/', max_len=12, seq_col='mutant', mut_col='mut_fn', wt_col='wt_fn',
                 target_col='target', pad_scale=0, positional_encoding=False, ic_weighted=False, mask=False,
                 feature_cols=None):
        super(ESMDataset, self).__init__()
        df['len'] = df[seq_col].apply(len)
        if feature_cols is None:
            feature_cols = []

        x_mut = torch.stack([pad_tensor(torch.load(f'{datapath}{x}')['representations'][33], max_len, pad_scale) for x in
                             df[mut_col].values], dim=0)
        x_wt = torch.stack([pad_tensor(torch.load(f'{datapath}{x}')['representations'][33], max_len, pad_scale) for x in
                            df[wt_col].values], dim=0)
        if ic_weighted:
            ic_weights = torch.from_numpy(get_ic_weights(df, ics_dict['KL'], max_len, seq_col, hla_col='HLA',
                                                         mask=mask, invert=False, threshold=0.2))[:,:,0:1] \
                              .repeat((1, 1, x_mut.shape[-1]))
            x_mut = (x_mut * ic_weights).float()
            x_wt = (x_wt * ic_weights).float()

        self.positional_encoding = positional_encoding
        self.pe_mut = get_positional_encoding(x_mut, pad_scale) if positional_encoding else None
        self.pe_wt = get_positional_encoding(x_wt, pad_scale) if positional_encoding else None

        # Mask for padding ; Allows to standardize the sequence features without considering the pads
        # TODO: Is this behaviour actually correct?
        # Creating the mask to allow selection of kmers without padding
        # Here, using a single mask because in theory WT and MUT are aligned and have the same length thus same pad
        x_mask = torch.from_numpy(df['len'].values) - 1
        range_tensor = torch.arange(max_len).unsqueeze(0).repeat(len(x_mut), 1)
        self.x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)
        self.x_mut = x_mut.contiguous()
        self.x_wt = x_wt.contiguous()
        self.y = y.contiguous()

        if len(feature_cols) > 0:
            self.x_features = torch.from_numpy(df[feature_cols].values).float()
            self.extra_features_flag = True
        else:
            self.x_features = torch.empty((len(x_mut),))
            self.extra_features_flag = False

        self.df = df
        self.len = len(y)
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Returns the appropriate input tensors (X_mut, x_wt, y)
        :param idx:
        :return:
        """

        if self.extra_features_flag:
            return self.x_mut[idx], self.x_wt[idx], self.x_features[idx], self.y[idx]
        else:
            return self.x_mut[idx], self.x_wt[idx], self.y[idx]

    def get_std_tensors(self):
        if self.extra_features_flag:
            return self.x_mut, self.x_wt, self.x_mask, self.x_features
        else:
            return self.x_mut, self.x_wt, self.x_mask


def get_esm_dataloader(df, datapath, max_len, seq_col='mutant', mut_col='mut_fn', wt_col='wt_fn', target_col='target',
                       pad_scale=0, positional_encoding=False, ic_weighted=False, mask=False, feature_cols=None,
                       batch_size=64, sampler=torch.utils.data.RandomSampler, return_dataset=True):
    dataset = ESMDataset(df, datapath, max_len, seq_col, mut_col=mut_col, wt_col=wt_col, target_col=target_col,
                         pad_scale=pad_scale, positional_encoding=positional_encoding, ic_weighted=ic_weighted, mask=mask,
                         feature_cols=feature_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))

    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader


# NNAlign stuff


class NNAlignDataset(Dataset):
    """
    Here for now, only get encoding and try to
    """

    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target',
                 pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW',
                 feature_cols: list = None):

        super(NNAlignDataset, self).__init__()
        # Encoding stuff
        if feature_cols is None:
            feature_cols = []
        df['len'] = df[seq_col].apply(len)
        df = df.query('len<=@max_len')
        matrix_dim = 21 if indel else 20

        # TODO: Implement the IC weights stuff at some point here
        #       to scale inputs for NNAlign with encode_batch_weighted()
        x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)

        # Creating the mask to allow selection of kmers without padding
        x_mask = torch.from_numpy(df['len'].values) - window_size
        range_tensor = torch.arange(max_len - window_size + 1).unsqueeze(0).repeat(len(x), 1)
        # Mask for Kmers + padding
        self.x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        # Creating another mask for the burn-in period+bool flag switch
        self.burn_in_mask = _get_burnin_mask_batch(df[seq_col].values, max_len, window_size, burnin_alphabet).unsqueeze(
            -1)
        self.burn_in_flag = False
        # Expand and unfold the sub kmers and the target to match the shape ; contiguous to allow for view operations
        self.x_tensor = x.unfold(1, window_size, 1).transpose(2, 3) \
            .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim).flatten(2, 3).contiguous()
        self.y = y.contiguous()

        # Add extra features
        if len(feature_cols) > 0:
            self.x_features = torch.from_numpy(df[feature_cols].values).float()
            self.extra_features_flag = True
        else:
            self.x_features = torch.empty((len(x),))
            self.extra_features_flag = False

        # Saving df in case it's needed
        self.df = df
        self.len = len(x)
        self.max_len = max_len
        self.seq_col = seq_col
        self.window_size = window_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Returns the appropriate input tensors (X, ..., y) depending on the bool flags
        A bit convoluted return, but basically 4 conditions:
            1. No burn-in, no extra features --> returns the normal x_tensor, kmers mask, target
            2. Burn-in, no extra features --> returns the normal x_tensor, burn-in mask, target
            3. No Burn-in, + extra features --> returns the normal x_tensor, kmers mask, x_features, target
            4. Burn-in, + extra features --> returns the normal x_tensor, burn-in mask, x_features, target
        :param idx:
        :return:
        """
        if self.burn_in_flag:
            if self.extra_features_flag:
                # 4
                return self.x_tensor[idx], self.burn_in_mask[idx], self.x_features[idx], self.y[idx]
            else:
                # 2
                return self.x_tensor[idx], self.burn_in_mask[idx], self.y[idx]
        else:
            if self.extra_features_flag:
                # 3
                return self.x_tensor[idx], self.x_mask[idx], self.x_features[idx], self.y[idx]
            else:
                # 1
                return self.x_tensor[idx], self.x_mask[idx], self.y[idx]

    def burn_in(self, flag):
        self.burn_in_flag = flag


def get_NNAlign_dataloader(df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                           seq_col: str = 'Peptide', target_col: str = 'agg_label',
                           pad_scale: float = None, indel: bool = False,
                           burnin_alphabet: str = 'ILVMFYW', feature_cols: list = None, batch_size=64,
                           sampler=torch.utils.data.RandomSampler,
                           return_dataset=False):
    dataset = NNAlignDataset(df, max_len, window_size, encoding, seq_col, target_col, pad_scale, indel, burnin_alphabet,
                             feature_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader


def _get_burnin_mask_batch(sequences, max_len, motif_len, alphabet='ILVMFYW'):
    return torch.stack([_get_burnin_mask(x, max_len, motif_len, alphabet) for x in sequences])


def _get_burnin_mask(seq, max_len, motif_len, alphabet='ILVMFYW'):
    mask = torch.tensor([x in alphabet for i, x in enumerate(seq) if i < len(seq) - motif_len + 1]).float()
    return F.pad(mask, (0, (max_len - motif_len + 1) - len(mask)), 'constant', 0)
