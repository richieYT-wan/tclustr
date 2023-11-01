import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import wandb
import math
import torch
from torch import optim
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists
from src.torch_utils import load_checkpoint
from src.models import PairedFVAE
from src.train_eval import predict_model, train_eval_loops
from src.datasets import PairedDataset
from src.metrics import PairedVAELoss
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    """
    Data processing args
    """
    parser.add_argument('-logwb', '--log_wandb', dest='log_wandb', required=False, default=False,
                        type=str2bool, help='Whether to log a run using WandB. False by default')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/230927_nettcr_positives_only.csv',
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-cdr3b', '--cdr3b_col', dest='cdr3b_col', default='TRB_CDR3', type=str, required=False,
                        help='Name of the column containing CDR3b sequences (inputs). '
                             'if "None", use_b will be deactivated.')
    parser.add_argument('-cdr3a', '--cdr3a_col', dest='cdr3a_col', default='TRA_CDR3', type=str, required=False,
                        help='Name of the column containing CDR3a sequences (inputs). '
                             'if "None", use_a will be deactivated.')
    parser.add_argument('-pep', '--pep_col', dest='pep_col', default='peptide', type=str, required=False,
                        help='Name of the column containing peptide sequences (inputs). '
                             'if "None", use_pep will be deactivated.')
    parser.add_argument('-v', '--v_col', dest='v_col', default='TRBV_gene', type=str, required=False,
                        help='Name of the column containing V genes (inputs).' \
                             'Will be TRBV_gene by default. If "None", use_v will be deactivated.')
    parser.add_argument('-j', '--j_col', dest='j_col', default='TRBJ_gene', type=str, required=False,
                        help='Name of the column containing J genes (inputs).' \
                             'Will be TRBJ_gene by default. If "None", use_j will be deactivated.')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-mla', '--max_len_a', dest='max_len_a', type=int, required=True, default=24,
                        help='Maximum CDR3A sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb', '--max_len_b', dest='max_len_b', type=int, required=True, default=25,
                        help='Maximum CDR3B sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlpep', '--max_len_pep', dest='max_len_pep', type=int, required=True, default=12,
                        help='Maximum peptide sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
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
    parser.add_argument('-lwb', '--weight_beta', dest='weight_beta', type=float, default=2,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwa', '--weight_alpha', dest='weight_alpha', type=float, default=2,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwp', '--weight_pep', dest='weight_pep', type=float, default=1,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld', '--weight_kld', dest='weight_kld', type=float, default=1,
                        help='Which weight to use for the KLD term in the loss')
    parser.add_argument('-lwv', '--weight_v', dest='weight_v', type=float, default=1,
                        help='Which weight to use for the V gene term in the loss')
    parser.add_argument('-lwj', '--weight_j', dest='weight_j', type=float, default=.75,
                        help='Which weight to use for the J gene term in the loss')
    parser.add_argument('-wu', '--warm_up', dest='warm_up', type=int, default=10,
                        help='Whether to do a warm-up period for the loss (without the KLD term). ' \
                             'Default = 10. Set to 0 if you want this disabled')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')

    # TODO: TBD what to do with these!
    """
    TODO: Misc. 
    """
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=13,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()


def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    torch.manual_seed(args['seed'])
    # Convert the activation string codes to their nn counterparts
    args['activation'] = {'selu': nn.SELU(), 'relu': nn.ReLU(),
                          'leakyrelu': nn.LeakyReLU(), 'elu': nn.ELU()}[args['activation']]

    # Loading data and getting train/valid
    # TODO: Restore valid kcv behaviour // or not
    df = pd.read_csv(args['file'])
    dfname = args['file'].split('/')[-1].split('.')[0]

    if args['fold'] is not None:
        train_df = df.query('partition!=@args["fold"]')
        valid_df = df.query('partition==@args["fold"]')
    else:
        train_df = df
        valid_df = df.query('partition==0')
    args['n_batches'] = math.ceil(len(train_df) / args['batch_size'])
    # TODO: get rid of this bad hardcoded behaviour for AA_dim ; Let's see if we end up using Xs
    args['aa_dim'] = 20
    args['use_a'] = not (args['cdr3a_col'] == "None")
    args['use_b'] = not (args['cdr3b_col'] == "None")
    args['use_pep'] = not (args['pep_col'] == "None")
    args['use_v'] = not (args['v_col'] == "None")
    args['use_j'] = not (args['j_col'] == "None")
    args['v_dim'] = 51
    args['j_dim'] = 13
    if args['log_wandb']:
        wandb.login()
    # File-saving stuff
    connector = '' if args["out"] == '' else '_'
    kf = '-1' if args["fold"] is None else args['fold']
    rid = args['random_id'] if (args['random_id'] is not None and args['random_id'] != '') else get_random_id() if args[
                                                                                                                       'random_id'] == '' else \
    args['random_id']
    unique_filename = f'{args["out"]}{connector}KFold_{kf}_{get_datetime_string()}_{rid}'

    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    outdir = os.path.join('../output/', unique_filename) + '/'
    mkdirs(outdir)

    # Def params so it's tidy

    # Maybe this is better? Defining the various keys using the constructor's init arguments
    model_init_code = PairedFVAE.__init__.__code__
    model_init_code = PairedFVAE.__init__.__code__.co_varnames[1:model_init_code.co_argcount]
    model_keys = [x for x in args.keys() if x in model_init_code]
    dataset_init_code = PairedDataset.__init__.__code__
    dataset_init_code = PairedDataset.__init__.__code__.co_varnames[1:dataset_init_code.co_argcount]
    dataset_keys = [x for x in args.keys() if x in dataset_init_code]
    loss_init_code = PairedVAELoss.__init__.__code__
    loss_init_code = PairedVAELoss.__init__.__code__.co_varnames[1:loss_init_code.co_argcount]
    loss_keys = [x for x in args.keys() if x in loss_init_code]

    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    loss_params = {k: args[k] for k in loss_keys}
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    torch.manual_seed(args["fold"])
    # Here, don't specify V and J map to use the default V/J maps loaded from src.data_processing
    train_dataset = PairedDataset(train_df, **dataset_params)
    valid_dataset = PairedDataset(valid_df, **dataset_params)
    print(train_dataset.x.shape)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory
    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)

    fold_filename = f'kcv_{dfname}_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_fold{args["fold"]:02}_{fold_filename}.pt'

    # instantiate objects
    model = PairedFVAE(**model_params)
    criterion = PairedVAELoss(**loss_params)
    optimizer = optim.Adam(model.parameters(), **optim_params)

    # Adding the wandb watch statement ; Only add them in the script so that it never interferes anywhere in train_eval
    if args['log_wandb']:
        # wandb stuff
        wandb.init(project=unique_filename, name=f'fold_{args["fold"]:02}', config=args)
        wandb.watch(model, criterion=criterion, log_freq=len(train_loader))

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_metrics = train_eval_loops(args['n_epochs'], args['tolerance'], model,
                                                                   criterion, optimizer, train_loader, valid_loader,
                                                                   checkpoint_filename, outdir)

    # Convert list of dicts to dicts of lists
    train_losses_dict = get_dict_of_lists(train_losses,
                                          'train')  # {f'train_{key}': [d[key] for d in train_losses] for key in train_losses[0]}
    train_metrics_dict = get_dict_of_lists(train_metrics,
                                           'train')  # {f'train_{key}': [d[key] for d in train_metrics] for key in train_metrics[0]}
    valid_losses_dict = get_dict_of_lists(valid_losses,
                                          'valid')  # {f'valid_{key}': [d[key] for d in valid_losses] for key in valid_losses[0]}
    valid_metrics_dict = get_dict_of_lists(valid_metrics,
                                           'valid')  # {f'valid_{key}': [d[key] for d in valid_metrics] for key in valid_metrics[0]}

    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}

    # Saving text file for the run:
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Fold: {args["fold"]}\n')
        file.write(f"Best valid epoch: {best_epoch}\n")
        for k, v in best_val_loss.items():
            file.write(f'{k}:\t{v}\n')
        for k, v in best_val_metrics.items():
            file.write(f'{k}:\t{v}\n')

    pkl_dump(train_losses_dict, f'{outdir}/train_losses_{fold_filename}.pkl')
    pkl_dump(valid_losses_dict, f'{outdir}/valid_losses_{fold_filename}.pkl')
    pkl_dump(train_metrics_dict, f'{outdir}/train_metrics_{fold_filename}.pkl')
    pkl_dump(valid_metrics_dict, f'{outdir}/valid_metrics_{fold_filename}.pkl')
    plot_vae_loss_accs(losses_dict, accs_dict, unique_filename, outdir,
                       dpi=300, palette='gnuplot2_r', warm_up=args['warm_up'])

    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=checkpoint_filename,
                            dir_path=outdir)
    valid_preds = predict_model(model, valid_dataset, valid_loader)
    valid_preds['fold'] = args["fold"]
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{fold_filename}.csv', index=False)
    acc_cols = [x for x in valid_preds.columns if '_acc' in x]
    valid_accs = {k: valid_preds[k].mean() for k in acc_cols}
    valid_accs['v_acc'] = valid_preds['v_correct'].mean() if args['use_v'] else None
    valid_accs['j_acc'] = valid_preds['j_correct'].mean() if args['use_j'] else None
    accs_text = 'Accs:\t' + ',\t'.join(
        [f"{k.replace('accuracy', 'acc')}:{v:.2%}" for k, v in valid_accs.items() if v is not None])
    valid_text = 'Final valid mean accuracies' + accs_text
    print(valid_text)
    if args['test_file'] is not None:
        test_df = pd.read_csv(args['test_file'])
        test_basename = os.path.basename(args['test_file']).split(".")[0]
        test_dataset = PairedDataset(test_df, v_map=train_dataset.v_map, j_map=train_dataset.j_map, **dataset_params)
        test_loader = test_dataset.get_dataloader(batch_size=3 * args['batch_size'],
                                                  sampler=SequentialSampler)

        test_preds = predict_model(model, test_dataset, test_loader)
        test_preds['fold'] = args["fold"]
        test_preds.to_csv(f'{outdir}test_predictions_{test_basename}_{fold_filename}.csv', index=False)
        acc_cols = [x for x in test_preds.columns if '_acc' in x]
        test_accs = {k: test_preds[k].mean() for k in acc_cols}
        test_accs['v_acc'] = test_preds['v_correct'].mean() if args['use_v'] else None
        test_accs['j_acc'] = test_preds['j_correct'].mean() if args['use_j'] else None
        accs_text = 'Accs:\t' + ',\t'.join(
            [f"{k.replace('accuracy', 'acc')}:{v:.2%}" for k, v in test_accs.items() if v is not None])
        test_text = 'Final valid mean accuracies' + accs_text
        print(test_text)
    else:
        test_accs = {}

    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Fold: {kf}')
        for k,v in valid_accs.items():
            file.write(f'Best valid {k}: {v}\n')
        if args['test_file'] is not None:
            for k, v in test_accs.items():
                file.write(f'Best test {k}: {v}\n')
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
