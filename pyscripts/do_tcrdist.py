import pwseqdist as pw
import os
import pandas as pd
import numpy as np
import argparse
from tcrdist.rep_funcs import _pws, _pw
from pathlib import Path


def do_tcrdist3_pipeline(df, args,
                         vdist=pw.metrics.nb_vector_tcrdist,
                         dmatrix=pw.matrices.tcr_nb_distance_matrix,
                         ctrim=0,
                         ntrim=0, ):
    df = df.copy()
    df.rename(columns={args['a1_col']: 'cdr1_a_aa', args['a2_col']: 'cdr2_a_aa', args['a3_col']: 'cdr3_a_aa',
                       args['b1_col']: 'cdr1_b_aa', args['b2_col']: 'cdr2_b_aa', args['b3_col']: 'cdr3_b_aa',
                       args['pep_col']: 'epitope'}, inplace=True)
    metrics_a = {'cdr1_a_aa': vdist, 'cdr2_a_aa': vdist, 'cdr3_a_aa': vdist}
    metrics_b = {'cdr1_b_aa': vdist, 'cdr2_b_aa': vdist, 'cdr3_b_aa': vdist}

    kargs_a = {'cdr1_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr2_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr3_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False}}

    kargs_b = {'cdr1_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr2_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr3_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False}}

    weights_a = {'cdr1_a_aa': 1, 'cdr2_a_aa': 1, 'cdr3_a_aa': 3}
    weights_b = {'cdr1_b_aa': 1, 'cdr2_b_aa': 1, 'cdr3_b_aa': 3}

    dmats_b = _pws(df=df,
                   metrics=metrics_b,
                   weights=weights_b,
                   kargs=kargs_b,
                   cpu=1,
                   uniquify=True,
                   store=True)

    dmats_a = _pws(df=df,
                   metrics=metrics_a,
                   weights=weights_a,
                   kargs=kargs_a,
                   cpu=1,
                   uniquify=True,
                   store=True)
    alpha_tcrdist = dmats_a['tcrdist']
    beta_tcrdist = dmats_b['tcrdist']
    ab_tcrdist: np.array = alpha_tcrdist + beta_tcrdist
    return ab_tcrdist


def do_tcrdist3_CDR3_pipeline(df, args,
                              vdist=pw.metrics.nb_vector_tcrdist,
                              dmatrix=pw.matrices.tcr_nb_distance_matrix,
                              ctrim=0,
                              ntrim=0, ):
    df = df.copy()
    df.rename(columns={args['a1_col']: 'cdr1_a_aa', args['a2_col']: 'cdr2_a_aa', args['a3_col']: 'cdr3_a_aa',
                       args['b1_col']: 'cdr1_b_aa', args['b2_col']: 'cdr2_b_aa', args['b3_col']: 'cdr3_b_aa',
                       args['pep_col']: 'epitope'}, inplace=True)
    metrics_a = {'cdr1_a_aa': vdist, 'cdr2_a_aa': vdist, 'cdr3_a_aa': vdist}
    metrics_b = {'cdr1_b_aa': vdist, 'cdr2_b_aa': vdist, 'cdr3_b_aa': vdist}

    kargs_a = {'cdr1_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr2_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr3_a_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False}}

    kargs_b = {'cdr1_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr2_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False},
               'cdr3_b_aa': {'use_numba': True, 'distance_matrix': dmatrix, 'ntrim': ntrim, 'ctrim': ctrim,
                             'fixed_gappos': False}}

    weights_a = {'cdr1_a_aa': 0, 'cdr2_a_aa': 0, 'cdr3_a_aa': 1}
    weights_b = {'cdr1_b_aa': 0, 'cdr2_b_aa': 0, 'cdr3_b_aa': 1}

    dmats_b = _pws(df=df,
                   metrics=metrics_b,
                   weights=weights_b,
                   kargs=kargs_b,
                   cpu=1,
                   uniquify=True,
                   store=True)

    dmats_a = _pws(df=df,
                   metrics=metrics_a,
                   weights=weights_a,
                   kargs=kargs_a,
                   cpu=1,
                   uniquify=True,
                   store=True)
    alpha_tcrdist = dmats_a['tcrdist']
    beta_tcrdist = dmats_b['tcrdist']
    ab_tcrdist: np.array = alpha_tcrdist + beta_tcrdist

    return ab_tcrdist


def args_parser():
    parser = argparse.ArgumentParser(
        description='Script to load a VAE model, extract similarity (or dist) and do TCRbase')
    """
    Data processing args
    """
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default=None, help='filepath of the input reference file')
    parser.add_argument('-fmt', '--format', dest='format', required=False, default=',',
                        help='Separator to use for the input df ; by default, ",". (ex use "\\t" if tsv)')
    parser.add_argument('-ch', '--chains', dest='chains', default='full',
                        help='Whether to use all chains ("full") or CDR3s ("CDR3")')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-od', '--outdir', dest='outdir', required=False,
                        type=str, default=None, help='Additional output directory')
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
    parser.add_argument('-pep', '--pep_col', dest='pep_col', default='peptide', type=str, required=False,
                        help='Name of the column containing peptide sequences (inputs)')
    parser.add_argument('-idx', '--index_col', dest='index_col', default=None, type=str,
                        help='Name of an index column to store and combine the df and the array (None by default, leave blank if no specific index column used)')
    parser.add_argument('-others', '--other_cols', dest='others', type=str, nargs='*', help='Other columns to add')
    return parser.parse_args()


def main():
    args = vars(args_parser())
    print(args)
    # Read and run tcrdist3
    df = pd.read_csv(args['file'], sep=args['format'])
    file_extension = Path(args['file']).suffix
    assert args['chains'] in ['full', 'CDR3'], f"chains must be either 'full' or 'CDR3'; got {args['chains']} instead!"
    print('Running TCRdist3')
    f = do_tcrdist3_pipeline if args['chains'] == 'full' else do_tcrdist3_CDR3_pipeline
    output_array = f(df, args)
    # Creating output and adding the extra columns
    output_df = pd.DataFrame(output_array)
    if args['index_col'] is None:
        print('No index col included. Re-saving input with an extra seq_id column.')
        df['seq_id'] = [f'seq_{i:05}' for i in range(len(df))]
        output_df['seq_id'] = df['seq_id']
        args['index_col'] = ['seq_id']
        df.to_csv(args['file'])
    else:
        output_df[args['index_col']] = df[args['index_col']]

    output_df[args['pep_col']] = df[args['pep_col']]
    if args['others'] is not None:
        output_df[args['others']] = df[args['others']]
    # Saving and filepath stuff
    out_fn = os.path.basename(args['file'].split('/')[-1].replace(file_extension, ''))
    out_fn = f'{out_fn}_tcrdist3_distmatrix'
    if not (args['out'] is None or len(args['out']) == 0):
        out_fn = f'{out_fn}_{args["out"]}'

    outdir = '../output/'
    if args['outdir'] is not None:
        outdir = os.path.join(outdir, args['outdir'])
        if not outdir.endswith('/'):
            outdir = outdir + '/'

    output_df.to_csv(f'{outdir}{out_fn}.txt')
    print(f'Output saved at {outdir}{out_fn}.txt')


if __name__ == '__main__':
    main()
