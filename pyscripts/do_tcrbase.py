import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import wandb
import math
import torch
import json
from torch import optim
from torch import cuda
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists
from src.torch_utils import load_checkpoint
from src.models import FullTCRVAE
from src.train_eval import predict_model, train_eval_loops
from src.datasets import TCRSpecificDataset
from src.metrics import CombinedVAELoss, get_metrics
from sklearn.metrics import roc_auc_score, precision_score
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-model_checkpoint', dest='model_checkpoint', required=True, type=str,
                        help='Path to the checkpoint (model weight) to load')
    parser.add_argument('-model_kwargs', dest='model_kwargs', required=True, type=str,
                        help='Path to the JSON containing the information necessary to instantiate the correct model. Has to be a JSON!')
    parser.add_argument('-dataset_kwargs', dest='dataset_kwargs', required=True, type=str,
                        help='Path to the JSON containing the information necessary to instantiate the dataset object. Has to be a JSON!')
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int,
                        help='K in K-fold. If used, will disregard the train/test file and try to query a "partition" column in the trainfile (-f) to split the train and validation data to be used')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/230921_nettcr_immrepnegs_noswap.csv',
                        help='filename of the input data file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str, required=True,
                        help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-a1', '--a1_col', dest='a1_col', default='A1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a2', '--a2_col', dest='a2_col', default='A2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a3', '--a3_col', dest='a3_col', default='A3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b1', '--b1_col', dest='b1_col', default='B1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b2', '--b2_col', dest='b2_col', default='B2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b3', '--b3_col', dest='b3_col', default='B3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=8,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=6,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb3', '--max_len_b3', dest='max_len_b3', type=int, default=23,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')

    """
    Models args 
    """
    parser.add_argument('-nh', '--hidden_dim', dest='hidden_dim', type=int, default=256,
                        help='Number of hidden units in the VAE. Default = 256')
    parser.add_argument('-nl', '--latent_dim', dest='latent_dim', type=int, default=128,
                        help='Size of the latent dimension. Default = 128')
    parser.add_argument('-act', '--activation', dest='activation', type=str, default='selu',
                        help='Which activation to use. Will map the correct nn.Module for the following keys:' \
                             '[selu, relu, leakyrelu, elu]')

    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=5e-4, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=256, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=2000, required=False,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=1,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld', '--weight_kld', dest='weight_kld', type=float, default=1e-1,
                        help='Which weight to use for the KLD term in the loss')
    parser.add_argument('-lwvae', '--weight_vae', dest='weight_vae', default=1,
                        help='Weight for the VAE term (reconstruction+KLD)')
    parser.add_argument('-lwtrp', '--weight_triplet',
                        dest='weight_triplet', type=float, default=1, help='Weight for the triplet loss term')
    parser.add_argument('-dist_type', '--dist_type', dest='dist_type', default='cosine', type=str,
                        help='Which distance metric to use ')
    parser.add_argument('-margin', dest='margin', default=None, type=float,
                        help='Margin for the triplet loss (Default is None and will have the default behaviour depending on the distance type)')
    parser.add_argument('-wu', '--warm_up', dest='warm_up', type=int, default=10,
                        help='Whether to do a warm-up period for the loss (without the KLD term). ' \
                             'Default = 10. Set to 0 if you want this disabled')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')

    # TODO: TBD what to do with these!
    """
    TODO: Misc. 
    """
    # These two arguments are to be phased out or re-used, in the case of fold.
    # For now, for the exercise, I will do KCV and try to see if there is any robustsness across folds in the VAE
    # later on, it makes no sense to concatenate the latent dimensions so we need to figure something else out.
    # parser.add_argument('-s', '--split', dest='split', required=False, type=int,
    #                     default=5, help='How to split the train/test data (test size=1/X)')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
