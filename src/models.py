import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_processing import get_positional_encoding
import math
#import wandb

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


class LSTMModule(NetParent):

    def __init__(self, input_dim=20, hidden_dim=64, num_layers=1, bidirectional=False):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    # TODO:
    #   Should check whether it makes more sense to put this here in the final module that combines everything.
    #   In theory shouldn't change anything, but gotta check out in practice
    #   In this module, I will only return out_mut, out_wt, to keep all information.
    #   In subsequent modules, I can then decide how to take care of the aggregation and representation
    #   This way, the code is more modular and LSTMModule only does pure LSTM
    def forward(self, x):
        # output should have shape (N, len, hidden)
        # z = output, (hidden, cell)
        out, (hidden, cell) = self.lstm(x)
        return out, hidden.permute(1, 0, 2).flatten(start_dim=1), cell.permute(1, 0, 2).flatten(start_dim=1)


class AttentionModule(NetParent):
    # TODO : test, either as a aggregation/pooling method or as a module replacing LSTM
    def __init__(self, input_dim, pad_scale, num_heads=2, dropout=0.):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        # self.max_len = max_len
        # self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.pad_scale = pad_scale
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout, batch_first=True)

    def forward(self, x, need_weights=False):
        # Explicitly takes pe here to make my life a bit easier during dataset/data processing and forward calls
        # in a lower / next module
        # because we need to standardize the data BEFORE adding the positional encoding
        attention_output, attention_weight = self.attention(x, x, x, need_weights=need_weights)

        return attention_output, attention_weight


# TODO : deal with this stupid pooling thing ; For now, just check that the concat sandwich works-ish
#   and gives some sort of minimal result (say 63-65% AUC ?)
class PoolingModule(NetParent):

    def __init__(self, pool_type='max', **kwargs):
        super(PoolingModule, self).__init__()
        pool = {'max': nn.MaxPool2d,
                'mean': nn.AvgPool2d,
                'global_max': nn.AdaptiveMaxPool1d,
                'global_mean': nn.AdaptiveAvgPool1d}
        # TODO: Forget attention_module for now until I understand better what it does / how it works
        # 'attention_module': nn.MultiheadAttention}
        self.pool = pool[pool_type](**kwargs)

    def forward(self, x):
        return self.pool(x)


class AggregatingPoolingModule(NetParent):

    def __init__(self, in_dim, pool_type, pool_kwargs=None, to_pool=True, agg_before_pooling=False):
        super(AggregatingPoolingModule, self).__init__()
        if pool_kwargs is None:
            pool_kwargs = {}
        self.pooling = PoolingModule(pool_type, **pool_kwargs) if to_pool else nn.Identity()
        self.agg_before_pooling = agg_before_pooling
        self.pool_type = pool_type

    def forward(self, z_mut, z_wt):
        # Flattening in case there's an extra batch dimension
        # z_mut, z_wt = z_mut.squeeze(0), z_wt.squeeze(0)

        # self.pooling will be bypassed with identity if self.to_pool is not true
        if self.agg_before_pool:
            z = torch.cat([z_mut, z_wt], dim=1)
            z = self.pooling(z)
        else:
            z_mut = self.pooling(z_mut)
            z_wt = self.pooling(z_wt)
            z = torch.cat([z_mut, z_wt], dim=1)

        return z


class PredictorModule(NetParent):
    """
    Learning from my mistake (NNAlignEF reloading was not working properly compared to EF2)
    So here, I split the predictor into its own class for the final FC layer(s) prediction module,
    Then I update the state_dict function so that I can re-load everything more easily/correctly
    without having to use strict=False which might cause a lot of errors to appear
    """

    # noinspection PyTypeChecker
    def __init__(self, input_dim, hidden_dim, num_layers=1, activation=nn.ReLU(),
                 dropout=0., batchnorm=False):
        super(PredictorModule, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), activation,
                  nn.Dropout(dropout)] if batchnorm \
            else [nn.Linear(input_dim, hidden_dim), activation, nn.Dropout(dropout)]
        for n in range(num_layers - 1):

            layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
            hidden_dim = hidden_dim // 2
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

        final_layer = [nn.Linear(hidden_dim, 1)]
        layers.extend(final_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SandwichLSTM(NetParent):
    # TODO:
    #   Here, should at some point replace input_dim, hidden_dim, num_layers to a input_network_kwargs
    #   And implement a dictionary to call another constructor (for example if we use CNN instead of LSTM
    #   Then we would have, instead of self.lstm_module : self.encoder (that is either LSTM, CNN, or something else)
    #   Then the pooling etc treats the output of Z the same. So maybe self.representation should be put in lstmmodule
    def __init__(self, input_dim, n_hidden_encoder=64, n_layers_encoder=1, bidirectional=False, n_hidden_predictor=30,
                 n_layers_predictor=1, predictor_activation=nn.ReLU(), dropout=0, batchnorm_predictor=False,
                 standardize=False, pool_type='max', pool_kwargs={'kernel_size': 1}, to_pool=False,
                 agg_before_pooling=False, representation='hidden', n_extrafeatures=0, standardize_features=True):
        super(SandwichLSTM, self).__init__()

        # TODO 230817: TESTING VECTORIZED STANDARDIZER INSTEAD OF 20 AA STD ; Works better (68% AUC on valid vs ~66%)
        self.standardizer_mut = StandardizerSequenceVector(input_dim, max_len=12) if standardize else StdBypass()
        self.standardizer_wt = StandardizerSequenceVector(input_dim, max_len=12) if standardize else StdBypass()

        # This is added so we can add extra features such as the %Rank to the sequences
        self.n_extrafeatures = n_extrafeatures
        if n_extrafeatures > 0:
            self.standardizer_features = StandardizerFeatures(
                n_feats=n_extrafeatures) if standardize_features else StdBypass()
        # TODO : add EF and all that pertains to it incl. EF standardizer etc

        self.lstm = LSTMModule(input_dim, n_hidden_encoder, n_layers_encoder, bidirectional)
        self.init_params = {key: value for key, value in locals().items() if key != 'self' and key != '__class__'}

        # # TODO: To fix code
        # factor = num_layers if representation in ['hidden', 'cell'] else 1
        #
        # self.agg_pooling = AggregatingPoolingModule(in_dim=hidden_dim * factor, pool_type=pool_type,
        #                                             pool_kwargs=pool_kwargs,
        #                                             to_pool=to_pool, agg_before_pooling=agg_before_pooling)

        self.indexing = {'ts': 0, 'hidden': 1, 'cell': 2}[representation]
        self.representation = representation
        # TODO: get concatenated dim after pooling somewhere
        n_dir = 2 if bidirectional else 1
        # TODO: Current input_dim assumes no pooling of any sort so we just concatenate the x_mut and x_wt
        #       vectors and have n_hidden * 2 * n_layers * n_directions
        #       Additionally : Need to add extra features dimensions into this predictor

        self.predictor = PredictorModule(input_dim=(n_dir * n_hidden_encoder * 2 * n_layers_encoder) + n_extrafeatures,
                                         hidden_dim=n_hidden_predictor, num_layers=n_layers_predictor,
                                         activation=predictor_activation, dropout=dropout, batchnorm=batchnorm_predictor)

    # Here include x_features=None as a bypass in case we don't use it so that it's not necessary to the call
    def forward(self, x_mut, x_wt, x_features=None):
        with torch.no_grad():
            x_mut = self.standardizer_mut(x_mut)
            x_wt = self.standardizer_wt(x_wt)

        z_mut = self.lstm(x_mut)
        z_wt = self.lstm(x_wt)
        # Index the wanted representation from the tuples
        z_mut, z_wt = z_mut[self.indexing], z_wt[self.indexing]
        if self.representation == 'ts':
            z_mut, z_wt = z_mut[:, -1, :], z_wt[:, -1, :]
        # Concat without pooling for now
        z = torch.cat([z_mut, z_wt], dim=1)
        if self.n_extrafeatures > 0:
            x_features = self.standardizer_features(x_features)
            z = torch.cat([z, x_features], dim=1)

        # z = self.agg_pooling(z_mut, z_wt)
        z = self.predictor(z)
        # Here z should be scores, then if wanted we can add a sigmoid fct somewhere
        return z

    def predict(self, x_mut, x_wt, x_features=None):
        with torch.no_grad():
            z = self.forward(x_mut, x_wt, x_features)
            return F.sigmoid(z)

    def fit_standardizer(self, x_mut, x_wt, x_mask, x_features=None):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer_mut.fit(x_mut, x_mask)
            self.standardizer_wt.fit(x_wt, x_mask)
            if self.n_extrafeatures > 0 and x_features is not None:
                self.standardizer_features.fit(x_features)


class SandwichAttnLSTM(NetParent):
    def __init__(self, input_dim=20, pad_scale=-15,
                 num_heads_attention=4, dropout_attention=0., concat_order='concat_first', add_pe=True,
                 n_hidden_encoder=64, n_layers_encoder=1, bidirectional=False,
                 n_hidden_predictor=30, n_layers_predictor=1, predictor_activation=nn.ReLU(), dropout_predictor=0,
                 batchnorm_predictor=False,
                 pool_type='max', standardize=False, pool_kwargs=None, to_pool=False,
                 agg_before_pooling=False, representation='hidden', n_extrafeatures=0, standardize_features=False):
        super(SandwichAttnLSTM, self).__init__()

        # Orders are :
        #   1. concat->attn->lstm_module
        #   2. attn->concat->lstm_module
        #   3. attn->lstm_module->concat

        assert concat_order in ['concat_first', 'after_attention', 'after_lstm']
        self.concat_order = concat_order
        self.pad_scale = pad_scale
        self.add_pe = add_pe
        if pool_kwargs is None:
            pool_kwargs = {'kernel_size': 1}
        self.standardizer_mut = StandardizerSequenceVector(input_dim, max_len=12) if standardize else StdBypass()
        self.standardizer_wt = StandardizerSequenceVector(input_dim, max_len=12) if standardize else StdBypass()

        # This is added, so we can add extra features such as the %Rank to the sequences
        self.n_extrafeatures = n_extrafeatures
        if n_extrafeatures > 0:
            self.standardizer_features = StandardizerFeatures(
                n_feats=n_extrafeatures) if standardize_features else StdBypass()
        # TODO : add EF and all that pertains to it incl. EF standardizer etc
        self.attention_module = AttentionModule(input_dim, pad_scale, num_heads_attention, dropout_attention)
        self.lstm_module = LSTMModule(input_dim, n_hidden_encoder, n_layers_encoder, bidirectional)

        self.init_params = {key: value for key, value in locals().items() if key != 'self' and key != '__class__'}

        self.indexing = {'ts': 0, 'hidden': 1, 'cell': 2}[representation]
        self.representation = representation
        # TODO: Current input_dim assumes no pooling of any sort so we just concatenate the x_mut and x_wt
        #       vectors and have n_hidden * n_cat * n_layers * n_directions (HERE N_CAT REPLACES THE *2 IN THE OTHER MODEL
        #       BECAUSE IF WE CONCAT LAST THEN WE HAVE TWICE THE LSTM DIM, otherwise we have them a single time)
        #       Additionally : Need to add extra features dimensions into this predictor
        dim_dir = 2 if bidirectional else 1
        dim_cat = 2 if concat_order == 'after_lstm' else 1
        self.predictor = PredictorModule(
            input_dim=(dim_dir * n_hidden_encoder * dim_cat * n_layers_encoder) + n_extrafeatures,
            hidden_dim=n_hidden_predictor, num_layers=n_layers_predictor,
            activation=predictor_activation, dropout=dropout_predictor, batchnorm=batchnorm_predictor)

    # Here include x_features=None as a bypass in case we don't use it so that it's not necessary to the call
    def forward(self, x_mut, x_wt, x_features=None, need_weights=False):
        """
        Here, there are three operations possible
            1. Concat->Attention->LSTM->Predictor
            2. Attention->Concat->LSTM->Predictor
            3. Attention->LSTM->Concat->Predictor

        Depending on the order of the operations, predictor's input dim has to be adjusted
        Also for each case, the positional encoding should be treated differently. Standardizer should always happen first
        This is the reason for the ugly triple if/else loop, to treat each concatenation strategy

        Args:
            x_mut: mutant input tensor
            x_wt: wild-type input tensor
            x_features: extra features if needed
            need_weights: Return attention weights (False by default)

        Returns: z: predictions (logits) tensor of dimension (batch_size, 1)

        """

        # Always standardize them separately (due to paddings)
        with torch.no_grad():
            # Get the positional encoding before standardizer, add them whenever it makes sense
            # because it takes pad_scale into account, otherwise it will not get the right PEs
            z_mut = self.standardizer_mut(x_mut)
            z_wt = self.standardizer_wt(x_wt)

        if self.concat_order == 'concat_first':
            z = torch.cat([z_mut, z_wt], dim=1)
            # Do it this way to get the correct positional encoding using x_mut/x_wt (i.e. not standardized) because of pad_scale
            z = z + self.get_positional_encoding(torch.concat([x_mut, x_wt], dim=1)) if self.add_pe else z
            z, attention_weights = self.attention_module(z, need_weights=need_weights)
            z = self.lstm(z)
        else:
            z_mut = z_mut + get_positional_encoding(x_mut, self.pad_scale) if self.add_pe else z_mut
            z_wt = z_wt + get_positional_encoding(x_wt, self.pad_scale) if self.add_pe else z_wt

            if self.concat_order == 'after_attention':
                z_mut, atn_mut = self.attention_module(z_mut, need_weights=need_weights)
                z_wt, atn_wt = self.attention_module(z_wt, need_weights=need_weights)
                z = torch.cat([z_mut, z_wt], dim=1)
                z = self.lstm(z)
            elif self.concat_order == 'after_lstm':  # Here, concat_order == 'after_lstm':
                z_mut, atn_mut = self.attention_module(z_mut, need_weights=need_weights)
                z_wt, atn_wt = self.attention_module(z_wt, need_weights=need_weights)
                z_mut = self.lstm(z_mut)
                z_wt = self.lstm(z_wt)
                z = torch.cat([z_mut, z_wt], dim=1)
            else:
                raise Exception(f'No proper concat_order provided. Current: {self.concat_order}')

        if self.n_extrafeatures > 0:
            z = torch.cat([z, x_features], dim=1)

        z = self.predictor(z)
        # Here z should be scores, then if wanted we can add a sigmoid fct somewhere
        return z

    def predict(self, x_mut, x_wt):
        with torch.no_grad():
            z = self.forward(x_mut, x_wt)
            return F.sigmoid(z)

    def fit_standardizer(self, x_mut, x_wt, x_mask, x_features=None):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer_mut.fit(x_mut, x_mask)
            self.standardizer_wt.fit(x_wt, x_mask)
            if self.n_extrafeatures > 0:
                self.standardizer_features.fit(x_features)

    def get_positional_encoding(self, x, n=10000):
        batch_size, max_seq_len, n_features = x.size()

        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        mask = (~(x == self.pad_scale).all(dim=2)).float()
        pos = pos * mask

        div_term = torch.exp(torch.arange(0, n_features, 2, dtype=torch.float32) * (-math.log(n) / n_features))

        pe = torch.zeros(batch_size, max_seq_len, n_features)
        pe[:, :, 0::2] = torch.sin(pos.unsqueeze(2) * div_term)
        pe[:, :, 1::2] = torch.cos(pos.unsqueeze(2) * div_term)
        return pe

    def lstm(self, z):
        """

        Args:
            z: input tensor
        Returns:
            z: output tensor after running lstm module and indexing for the representation (hidden, cell, timestep)
        """
        z = self.lstm_module(z)
        z = z[self.indexing]
        if self.representation == 'ts':
            z = z[:, -1, :]
        return z

############################
####   NNAlign stuff    ####
############################
#
# class NNAlignSinglePass(NetParent):
#     """
#     NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.
#     """
#
#     def __init__(self, n_hidden, window_size,
#                  activation, batchnorm=False,
#                  dropout=0.0, indel=False):
#         super(NNAlignSinglePass, self).__init__()
#         self.matrix_dim = 21 if indel else 20
#         self.window_size = window_size
#         self.n_hidden = n_hidden
#         self.in_layer = nn.Linear(self.window_size * self.matrix_dim, n_hidden)
#         self.out_layer = nn.Linear(n_hidden, 1)
#         self.batchnorm = batchnorm
#         if batchnorm:
#             self.bn1 = nn.BatchNorm1d(n_hidden)
#         self.dropout = nn.Dropout(p=dropout)
#         self.act = activation
#
#     def forward(self, x_tensor: torch.Tensor, x_mask: torch.tensor):
#
#         """
#         Single forward pass for layers + best score selection without w/o grad
#         Args:
#             x_mask:
#             x_tensor:
#
#         Returns:
#
#         """
#         # FIRST FORWARD PASS: best scoring selection, with no grad
#         z = self.in_layer(x_tensor)  # Inlayer
#         # Flip dimensions to allow for batchnorm then flip back
#         if self.batchnorm:
#             z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
#                 .view(-1, x_tensor.shape[1], self.n_hidden)
#         z = self.dropout(z)
#         z = self.act(z)
#         z = self.out_layer(z)  # Out Layer for prediction
#
#         # NNAlign selecting the max score here
#         with torch.no_grad():
#             # Here, use sigmoid to set values to 0,1 before masking
#             # only for index selection, Z will be returned as logits
#             max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
#
#         z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers
#         return z
#
#     def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
#         """Works like forward but also returns the index (for the motif selection/return)
#
#         This should be done with torch no_grad as this shouldn't be used during/for training
#         Also here does the sigmoid to return scores within [0, 1] on Z
#         Args:
#             x_tensor: torch.Tensor, the input tensor (i.e. encoded sequences)
#             x_mask: torch.Tensor, to mask padded positions
#         Returns:
#             z: torch.Tensor, the best scoring K-mer for each of the input in X
#             max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
#                      used to find the predicted core
#         """
#         with torch.no_grad():
#             z = self.in_layer(x_tensor)
#             if self.batchnorm:
#                 z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
#                     .view(-1, x_tensor.shape[1], self.n_hidden)
#             z = self.act(self.dropout(z))
#             z = self.out_layer(z)
#             # Do the same trick where the padded positions are removed prior to selecting index
#             max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
#             # Additionally run sigmoid on z so that it returns proba in range [0, 1]
#             z = F.sigmoid(torch.gather(z, 1, max_idx).squeeze(1))
#             return z, max_idx
#
#     def predict_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
#         """ QOL method to return the predictions without Sigmoid + return the indices
#         To be used elsewhere down the line (in EF model)
#
#         Args:
#             x_tensor:
#             x_mask:
#
#         Returns:
#
#         """
#         with torch.no_grad():
#             z = self.in_layer(x_tensor)
#             if self.batchnorm:
#                 z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
#                     .view(-1, x_tensor.shape[1], self.n_hidden)
#             z = self.act(self.dropout(z))
#             z = self.out_layer(z)
#             # Do the same trick where the padded positions are removed prior to selecting index
#             max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
#             # Additionally run sigmoid on z so that it returns proba in range [0, 1]
#             z = torch.gather(z, 1, max_idx).squeeze(1)
#             return z, max_idx
#
#
# class NNAlign(NetParent):
#     def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0, indel=False,
#                  standardize=True, **kwargs):
#         super(NNAlign, self).__init__()
#         self.nnalign = NNAlignSinglePass(n_hidden, window_size, activation, batchnorm, dropout, indel)
#         self.standardizer = StandardizerSequence() if standardize else StdBypass()
#         # Save here to make reloading a model potentially easier
#         self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
#                             'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel,
#                             'standardize': standardize}
#
#     def fit_standardizer(self, x_tensor: torch.Tensor, x_mask):
#         assert self.training, 'Must be in training mode to fit!'
#         with torch.no_grad():
#             self.standardizer.fit(x_tensor, x_mask)
#
#     def forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
#         with torch.no_grad():
#             x_tensor = self.standardizer(x_tensor)
#         x_tensor = self.nnalign(x_tensor, x_mask)
#         return x_tensor
#
#     def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
#         with torch.no_grad():
#             x_tensor = self.standardizer(x_tensor)
#             x_tensor, max_idx = self.nnalign.predict(x_tensor, x_mask)
#             return x_tensor, max_idx
#
#     def predict_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
#         with torch.no_grad():
#             x_tensor = self.standardizer(x_tensor)
#             x_tensor, max_idx = self.nnalign.predict_logits(x_tensor, x_mask)
#             return x_tensor, max_idx
#
#     def reset_parameters(self, **kwargs):
#         for child in self.children():
#             if hasattr(child, 'reset_parameters'):
#                 try:
#                     child.reset_parameters(**kwargs)
#                 except:
#                     print('here xd', child)
#
#     def state_dict(self, **kwargs):
#         state_dict = super(NNAlign, self).state_dict()
#         state_dict['nnalign'] = self.nnalign.state_dict()
#         state_dict['standardizer'] = self.standardizer.state_dict()
#         state_dict['init_params'] = self.init_params
#         return state_dict
#
#     def load_state_dict(self, state_dict, **kwargs):
#         self.nnalign.load_state_dict(state_dict['nnalign'])
#         self.standardizer.load_state_dict(state_dict['standardizer'])
#         self.init_params = state_dict['init_params']
#
#
# class ExtraLayerSingle(NetParent):
#
#     def __init__(self, n_input, n_hidden, activation=nn.SELU(), batchnorm=False, dropout=0.0):
#         super(ExtraLayerSingle, self).__init__()
#         self.n_input = n_input
#
#         # This here exists for compatibility issues. We don't actually use any hidden but a single in -> out layer
#         self.n_hidden = n_hidden
#         # These here are used to batchnorm and dropout the concatenated inputs, rather than an intermediate layer nodes
#         self.dropout = nn.Dropout(dropout)
#         if batchnorm:
#             self.bn1 = nn.BatchNorm1d(n_input)
#         self.batchnorm = batchnorm
#         # Also exists for compatibility...
#         self.act = activation
#         self.layer = nn.Linear(n_input, 1)
#
#     def forward(self, x_concat):
#         """ Assumes we give it the concatenated (dim=1) input
#         The input should be the concat'd tensor between the tensor logits returned by NNAlign and the standardized features
#         Args:
#             x_concat:
#
#         Returns:
#             z
#         """
#         if self.batchnorm:
#             x_concat = self.bn1(x_concat)
#         z = self.dropout(x_concat)
#         z = self.layer(z)
#         return z
#
#     def predict(self, x_concat):
#         """
#         Convoluted but exists for compatibility issues
#         Args:
#             x_concat:
#
#         Returns:
#
#         """
#         return F.sigmoid(self(x_concat))
#
#     def state_dict(self, **kwargs):
#         state_dict = super(ExtraLayerSingle, self).state_dict()
#         state_dict['n_input'] = self.n_input
#         state_dict['n_hidden'] = self.n_hidden
#         # state_dict['dropout'] = self.dropout.p
#         state_dict['batchnorm'] = self.batchnorm
#         state_dict['act'] = self.act
#         return state_dict
#
#     def load_state_dict(self, state_dict, **kwargs):
#         self.n_input = state_dict['n_input']
#         self.n_hidden = state_dict['n_hidden']
#         # self.dropout = state_dict['dropout']
#         self.batchnorm = state_dict['batchnorm']
#         self.act = state_dict['act']
#
#
# class ExtraLayerDouble(NetParent):
#     def __init__(self, n_input, n_hidden, activation=nn.SELU(), batchnorm=False, dropout=0.0):
#         super(ExtraLayerDouble, self).__init__()
#         self.n_input = n_input
#         self.n_hidden = n_hidden
#         self.act = activation
#         self.batchnorm = batchnorm
#         self.dropout = nn.Dropout(dropout)
#         if batchnorm:
#             self.bn1 = nn.BatchNorm1d(n_hidden)
#
#         self.in_layer = nn.Linear(self.n_input, n_hidden)
#         self.out_layer = nn.Linear(n_hidden, 1)
#
#     def forward(self, x_concat):
#         """ Assumes x_concat comes from the concatenation of the output of NNAlign and X_features, standardized or not
#         Args:
#             x_concat:
#
#         Returns:
#             z: The result of the layers
#         """
#         z = self.in_layer(x_concat)
#         if self.batchnorm:
#             z = self.bn1(z)
#         z = self.act(self.dropout(z))
#         z = self.out_layer(z)
#         return z
#
#     def predict(self, x_concat):
#         """ Exists for compatibility / cleaner code issues
#         Args:
#             x_concat: Same as above.
#         Returns:
#             z
#         """
#         return F.sigmoid(self(x_concat))
#
#     def state_dict(self, **kwargs):
#         state_dict = super(ExtraLayerDouble, self).state_dict()
#         state_dict['n_input'] = self.n_input
#         state_dict['n_hidden'] = self.n_hidden
#         # state_dict['dropout'] = self.dropout.p
#         state_dict['batchnorm'] = self.batchnorm
#         state_dict['act'] = self.act
#         return state_dict
#
#     def load_state_dict(self, state_dict, **kwargs):
#         self.n_input = state_dict['n_input']
#         self.n_hidden = state_dict['n_hidden']
#         # self.dropout = state_dict['dropout']
#         self.batchnorm = state_dict['batchnorm']
#         self.act = state_dict['act']
#
#
# class NNAlignEF(NetParent):
#     """ EF == ExtraFeatures
#     TODO: Currently assumes that I need an extra in_layer + an extra out_layer
#           Could also be changed to take a single extra layer of nn.Linear(1+n_extrafeatures, 1)
#           That takes as input the logits from NNAlign + the extra features and directly returns a score without 2 layers.
#           Can maybe write another class EFModel that just takes the ef_xx part here
#     """
#
#     def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0,
#                  indel=False, standardize=True,
#                  n_extrafeatures=0, n_hidden_ef=5, activation_ef=nn.SELU(), batchnorm_ef=False, dropout_ef=0.0,
#                  **kwargs):
#         super(NNAlignEF, self).__init__()
#         # NNAlign part
#         self.nnalign_model = NNAlign(n_hidden, window_size, activation, batchnorm, dropout, indel, standardize)
#         # Extra layer part
#         self.in_dim = n_extrafeatures + 1  # +1 because that's the dimension of the logit scores returned by NNAlign
#         self.ef_standardizer = StandardizerFeatures() if standardize else StdBypass()
#         self.ef_inlayer = nn.Linear(self.in_dim, n_hidden_ef)
#         self.ef_outlayer = nn.Linear(n_hidden_ef, 1)
#         self.ef_act = activation_ef
#         self.ef_dropout = nn.Dropout(dropout_ef)
#         self.ef_batchnorm = batchnorm_ef
#
#         # TODO : If this is switched to a single layer, then BatchNorm1d should be updated to nn.BatchNorm1d(self.in_dim)
#         if batchnorm_ef:
#             self.ef_bn1 = nn.BatchNorm1d(n_hidden_ef)
#
#         self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
#                             'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel, 'standardize': standardize,
#                             'n_extrafeatures': n_extrafeatures, 'n_hidden_ef': n_hidden_ef,
#                             'activation_ef': activation_ef,
#                             'batchnorm_ef': batchnorm_ef, 'dropout_ef': dropout_ef}
#
#     def fit_standardizer(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         self.nnalign_model.fit_standardizer(x_tensor, x_mask)
#         self.ef_standardizer.fit(x_features)
#
#     def forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         # NNAlign part
#         z = self.nnalign_model(x_tensor, x_mask)
#         # Extra features part, standardizes, concat
#         x_features = self.ef_standardizer(x_features)
#         z = torch.cat([z, x_features], dim=1)
#         # Standard NN stuff for the extra layers
#         z = self.ef_inlayer(z)
#         if self.ef_batchnorm:
#             z = self.ef_bn1(z)
#         z = self.ef_act(self.ef_dropout(z))
#         # Returning logits
#         z = self.ef_outlayer(z)
#         return z
#
#     def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         """ TODO: This is a bit convoluted and could be reworked to be more efficient
#                   Would probly require to modify the other classes a bit though
#
#         Args:
#             x_tensor:
#             x_mask:
#             x_features:
#
#         Returns:
#
#         """
#         with torch.no_grad():
#             # Return logits from nnalign model + max idx
#             z, max_idx = self.nnalign_model.predict_logits(x_tensor, x_mask)
#
#             # Standard NN stuff for the extra layers
#             x_features = self.ef_standardizer(x_features)
#             z = torch.cat([z, x_features], dim=1)
#             z = self.ef_inlayer(z)
#             if self.ef_batchnorm:
#                 z = self.ef_bn1(z)
#             z = self.ef_act(self.ef_dropout(z))
#             # Returning probs [0, 1]
#             z = F.sigmoid(self.ef_outlayer(z))
#             return z, max_idx
#
#     def state_dict(self, **kwargs):
#         state_dict = super(NNAlignEF, self).state_dict()
#         state_dict['nnalign_model'] = self.nnalign_model.state_dict()
#         state_dict['ef_standardizer'] = self.ef_standardizer.state_dict()
#         state_dict['init_params'] = self.init_params
#         return state_dict
#
#     def load_state_dict(self, state_dict: dict, **kwargs):
#         self.nnalign_model.load_state_dict(state_dict['nnalign_model'])
#         self.ef_standardizer.load_state_dict(state_dict['ef_standardizer'])
#         self.init_params = state_dict['init_params']
#         to_filter = ['nnalign_model', 'ef_standardizer', 'init_params']
#         custom_state_dict = {k: state_dict[k] for k in [k for k in state_dict.keys() if k not in to_filter]}
#         # strict = False allows the loading of only the base layers and ignore the errors but this is
#         # a massive source of problem maybe ??
#         super(NNAlignEF, self).load_state_dict(custom_state_dict, strict=False)
#
#
# class NNAlignEF2(NetParent):
#     """
#     This class here used to test the difference using the class ExtraLayerXX instead of spelling out the layers here.
#
#     """
#
#     def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0,
#                  indel=False, standardize=True,
#                  extra_layer='single',
#                  n_extrafeatures=0, n_hidden_ef=5, activation_ef=nn.SELU(), batchnorm_ef=False, dropout_ef=0.0,
#                  **kwargs):
#         super(NNAlignEF2, self).__init__()
#         # NNAlign part
#         self.nnalign_model = NNAlign(n_hidden, window_size, activation, batchnorm, dropout, indel, standardize)
#         # Extra layer part
#         self.in_dim = n_extrafeatures + 1  # +1 because that's the dimension of the logit scores returned by NNAlign
#         self.ef_standardizer = StandardizerFeatures() if standardize else StdBypass()
#         constructor = dict(single=ExtraLayerSingle, double=ExtraLayerDouble)[extra_layer]
#         self.ef_layer = constructor(n_input=n_extrafeatures + 1, n_hidden=n_hidden_ef,
#                                     activation=activation_ef, batchnorm=batchnorm_ef, dropout=dropout_ef)
#
#         self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
#                             'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel, 'standardize': standardize,
#                             'n_extrafeatures': n_extrafeatures, 'n_hidden_ef': n_hidden_ef,
#                             'activation_ef': activation_ef,
#                             'batchnorm_ef': batchnorm_ef, 'dropout_ef': dropout_ef}
#
#     def fit_standardizer(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         self.nnalign_model.fit_standardizer(x_tensor, x_mask)
#         self.ef_standardizer.fit(x_features)
#
#     def forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         # NNAlign part
#         z = self.nnalign_model(x_tensor, x_mask)
#         # Extra features part, standardizes, concat
#         x_features = self.ef_standardizer(x_features)
#         z = torch.cat([z, x_features], dim=1)
#         # Standard NN stuff for the extra layers
#         return self.ef_layer(z)
#
#     def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
#         """ TODO: This is a bit convoluted and could be reworked to be more efficient
#                   Would probly require to modify the other classes a bit though
#
#         Args:
#             x_tensor:
#             x_mask:
#             x_features:
#
#         Returns:
#
#         """
#         with torch.no_grad():
#             # Return logits from nnalign model + max idx
#             z, max_idx = self.nnalign_model.predict_logits(x_tensor, x_mask)
#
#             # Standard NN stuff for the extra layers
#             x_features = self.ef_standardizer(x_features)
#             z = torch.cat([z, x_features], dim=1)
#             z = self.ef_layer.predict(z)
#             return z, max_idx
#
#     def state_dict(self, **kwargs):
#         state_dict = super(NNAlignEF2, self).state_dict()
#         state_dict['nnalign_model'] = self.nnalign_model.state_dict()
#         state_dict['ef_standardizer'] = self.ef_standardizer.state_dict()
#         state_dict['ef_layer'] = self.ef_layer.state_dict()
#         state_dict['init_params'] = self.init_params
#         return state_dict
#
#     def load_state_dict(self, state_dict, **kwargs):
#         self.nnalign_model.load_state_dict(state_dict['nnalign_model'])
#         self.ef_standardizer.load_state_dict(state_dict['ef_standardizer'])
#         self.ef_layer.load_state_dict(state_dict['ef_layer'])
#         self.init_params = state_dict['init_params']
