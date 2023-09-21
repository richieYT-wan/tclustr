import pandas as pd
import os
from joblib import Parallel, delayed
from functools import partial
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    """
    Data processing args
    """
    parser.add_argument('-f', '--folder', dest='folder', required=False, default=False,
                        type=str, help='path to folder')
    parser.add_argument('-p', '--percentile', dest='percentile', type=float, default=5.)

    return parser.parse_args()


def read_filter_df(filename):
    try:
        df = pd.read_csv(filename, sep='\t').query('productive_frequency!="na" and amino_acid !="na"')
        df = df[['bio_identity', 'amino_acid', 'productive_frequency', 'v_gene', 'j_gene', 'v_family', 'j_family', 'v_resolved', 'j_resolved']]
        df['ID'] = filename
        df['productive_frequency'] = df['productive_frequency'].astype(float)
        return df
    except:
        return pd.DataFrame()


def main():
    # I like dictionary for args :-)
    # TODO : finish this script if needed...
    args = vars(args_parser())
    return 0


if __name__ == '__main__':
    main()
