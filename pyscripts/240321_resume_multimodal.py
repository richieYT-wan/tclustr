import pandas as pd
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import wandb
import torch
from torch import optim
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists, get_class_initcode_keys, make_filename
from src.torch_utils import load_checkpoint, save_model_full, load_model_full, get_available_device, \
    save_json, load_json
from src.multimodal_models import BSSVAE, JMVAE
from src.multimodal_train_eval import predict_multimodal, multimodal_train_eval_loops
from src.datasets import MultimodalPepTCRDataset
from src.multimodal_metrics import BSSVAELoss, JMVAELoss
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to resume training and evaluate a multimodal VAE model'\
                                                 'Most parameters / arguments are set to None and will only be used if explicitely set, '\
                                                 'otherwise it will use the ones loaded from the run_parameters.'\
                                                 'Dataset parameters and models parameters are removed ; '\
                                                 'KLD Warm-up will resume from the epoch counter')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-device', dest='device', default=None, type=str,
                        help='Specify a device (cpu, cuda:0, cuda:1)')
    parser.add_argument('-f', '--file', dest='file', required=False, type=str,
                        default=None,
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default=None, help='Additional output name')
    """
    Models args 
    """
    parser.add_argument('-model_folder', type=str, required=True, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory and load the "best" checkpoint'\
                             'Unless another -pt_file is provided')
    parser.add_argument('-pt_file', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=None, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=None, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=1024, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=None, required=True,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=None, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=None,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld_n', '--weight_kld_n', dest='weight_kld_n', type=float, default=1e-2,
                        help='Which weight to use for the KLD (normal) term in the loss')
    parser.add_argument('-lwkld_z', '--weight_kld_z', dest='weight_kld_z', type=float, default=1,
                        help='Which weight to use for the KLD (Latent) term in the loss')
    parser.add_argument('-addkldn', '--add_kld_n_marg', dest='add_kld_n_marg', type=str2bool, default=False,
                        help='Add one more KLD term from Z_marg to N(0,1)')

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-pepweight', dest='pep_weighted', type=str2bool, default=False,
                        help='Using per-sample (by peptide label) weighted loss')
    # TODO: TBD what to do with these!
    """
    TODO: Misc. 
    """
    # These two arguments are to be phased out or re-used, in the case of fold.
    # For now, for the exercise, I will do KCV and try to see if there is any robustsness across folds in the VAE
    # later on, it makes no sense to concatenate the latent dimensions so we need to figure something else out.
    # parser.add_argument('-s', '--split', dest='split', required=False, type=int,
    #                     default=5, help='How to split the train/test data (test size=1/X)')
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()
