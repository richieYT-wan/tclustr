import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import argparse
from torch import nn
from torch.utils.data import SequentialSampler
from src.torch_utils import load_checkpoint
from src.models import CDR3bVAE
from src.train_eval import predict_model
from src.datasets import CDR3BetaDataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns


def args_parser():
    parser = argparse.ArgumentParser(
        description='Do distance extraction and clustering for all folds contained in a maindirectory')
    """
    Data processing args
    """
    parser.add_argument('-d', '--dir', dest='dir', required=False,
                        default='/home/projects/vaccine/people/yatwan/tclustr/output/30K_epochs_OnlyPositivesFullCDR3b_LowerDim_64_WD_1e-4_EDHpH/',
                        type=str, help='directory containing all 5 folds sub-directories')
    parser.add_argument('-f', '--file', dest='file', required=False,
                        default='/home/projects/vaccine/people/yatwan/tclustr/data/filtered/230927_nettcr_positives_only.csv',
                        type=str, help='train file')
    parser.add_argument('-o', '--outdir', dest='outdir', required=False,
                        type=str, default='/home/projects/vaccine/people/yatwan/tclustr/output/231012_redo_clusters/')

    return parser.parse_args()


def get_tcrbase_method(tcr, ref):
    # here take the top1 instead of percent
    best = ref[tcr].sort_values().head(1)
    best_name = best.index[0]
    best_sim = best.item()
    label = ref.loc[best_name]['labels']
    return label, best_name, best_sim


def main():
    args = vars(args_parser())
    maindir = args['dir']
    fold_dirs = sorted([f'{maindir}{subdir}/' for subdir in os.listdir(maindir)])
    for fdir in fold_dirs:
        distances = [f'{fdir}{x}' for x in os.listdir(fdir) if x.startswith('_') \
                     and x.endswith('csv') and 'dist' in x]
        assert len(distances)>0, 'ntr'
        print(fdir)
        with open(f'{args["outdir"]}tcrbase_results.txt', 'a') as f:
            f.write('#'*30)
            f.write('\n')
            f.write(f'{fdir}\n')
            f.write('#'*30)
            f.write('\n')
        for d in distances:
            best_dist = pd.read_csv(d, index_col=0)
            train_ref = best_dist.query('set=="train" and labels == "GILGFVFTL"')
            valid_query = best_dist.query('set=="valid"')
            valid_query = valid_query.drop(
                columns=[x for x in valid_query.columns if x != 'labels']).copy().reset_index().rename(
                columns={'index': 'CDR3b', 'labels': 'true_label'})
            valid_query[['similar_label', 'best_name', 'best_sim']] = valid_query.apply(
                lambda x: get_tcrbase_method(x['CDR3b'], ref=train_ref), axis=1, result_type='expand')
            valid_query['y_true'] = (valid_query['true_label'] == "GILGFVFTL").astype(int)

            auc = roc_auc_score(valid_query['y_true'], 1 - valid_query['best_sim'])
            auc01 = roc_auc_score(valid_query['y_true'], 1 - valid_query['best_sim'], max_fpr=0.1)
            print(f'{d}: {auc:.2%}, {auc01:.2%}')
            with open(f'{args["outdir"]}tcrbase_results.txt', 'a') as f:
                f.write(f'{d}: {auc:.2%}, {auc01:.2%}')



if __name__ == '__main__':
    main()
