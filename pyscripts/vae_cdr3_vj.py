import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import wandb
import copy
import torch
from torch import optim
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, \
    plot_loss_aucs, plot_vae_loss_accs
from src.torch_utils import load_checkpoint
from src.models import FullFVAE
from src.train_eval import predict_model, train_eval_loops
from src.datasets import CDR3BetaDataset
from src.metrics import VAELoss, get_metrics
from sklearn.metrics import roc_auc_score, precision_score
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    """
    Data processing args
    """
    parser.add_argument('-logwb', '--log_wandb', dest='log_wandb', required=False, default=False,
                        type=str2bool, help='Whether to log a run using WandB. False by default')
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/filtered/230921_nettcr_immrepnegs_noswap.csv',
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-cdr3', '--cdr3_col', dest='cdr3_col', default='B3', type=str, required=False,
                        help='Name of the column containing CDR3b sequences (inputs)')
    parser.add_argument('-v', '--v_col', dest='v_col', default='TRBV_gene', type=str, required=False,
                        help='Name of the column containing V genes (inputs).' \
                             'Will be TRBV_gene by default. If "None", use_v will be deactivated.')
    parser.add_argument('-j', '--j_col', dest='j_col', default='TRBJ_gene', type=str, required=False,
                        help='Name of the column containing J genes (inputs).' \
                             'Will be TRBJ_gene by default. If "None", use_j will be deactivated.')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-ml', '--max_len', dest='max_len', type=int, required=True, default=23,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')

    """
    Models args 
    """
    parser.add_argument('-nh', '--hidden_dim', dest='hidden_dim', type=int, default=256,
                        help='Number of hidden units in the VAE. Default = 256')
    parser.add_argument('-ld', '--latent_dim', dest='latent_dim', type=int, default=128,
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
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=3,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwv', '--weight_v', dest='weight_v', type=float, default=2.5,
                        help='Which weight to use for the V gene term in the loss')
    parser.add_argument('-lwj', '--weight_j', dest='weight_j', type=float, default=2,
                        help='Which weight to use for the J gene term in the loss')
    parser.add_argument('-lwkld', '--weight_kld', dest='weight_kld', type=float, default=1,
                        help='Which weight to use for the KLD term in the loss')
    parser.add_argument('-wu', '--warm_up', dest='warm_up', type=str2bool, default=True,
                        help='Whether to do a warm-up for the loss (without the KLD term). Default = False')
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
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default='',
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
    train_df = df.query('partition!=@args["fold"]')
    valid_df = df.query('partition==@args["fold"]')
    # TODO: get rid of this bad hardcoded behaviour for AA_dim ; Let's see if we end up using Xs
    args['aa_dim'] = 20
    args['use_v'] = False if args['v_col'] == "None" else True
    args['use_j'] = False if args['j_col'] == "None" else True
    args['v_dim'] = len(train_df[args['v_col']].unique())
    args['j_dim'] = len(train_df[args['j_col']].unique())
    rid = get_random_id(5) if args["random_id"] == '' else args["random_id"]
    if args['log_wandb']:
        wandb.login()
    # File-saving stuff
    connector = '' if args["out"] == '' else '_'
    kf = '-1' if args["fold"] is None else args['fold']
    unique_filename = f'{args["out"]}{connector}KFold_{kf}_{get_datetime_string()}_{rid}'

    # checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    outdir = os.path.join('../output/', unique_filename) + '/'
    mkdirs(outdir)

    # Def params so it's tidy

    # Maybe this is better? Defining the various keys using the constructor's init arguments
    model_init_code = FullFVAE.__init__.__code__
    model_init_code = FullFVAE.__init__.__code__.co_varnames[1:model_init_code.co_argcount]
    model_keys = [x for x in args.keys() if x in model_init_code]
    dataset_init_code = CDR3BetaDataset.__init__.__code__
    dataset_init_code = CDR3BetaDataset.__init__.__code__.co_varnames[1:dataset_init_code.co_argcount]
    dataset_keys = [x for x in args.keys() if x in dataset_init_code]
    loss_init_code = VAELoss.__init__.__code__
    loss_init_code = VAELoss.__init__.__code__.co_varnames[1:loss_init_code.co_argcount]
    loss_keys = [x for x in args.keys() if x in loss_init_code]

    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    loss_params = {k: args[k] for k in loss_keys}
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    train_dataset = CDR3BetaDataset(train_df, **dataset_params, v_map=None, j_map=None)
    valid_dataset = CDR3BetaDataset(valid_df, **dataset_params, v_map=train_dataset.v_map, j_map=train_dataset.j_map)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory
    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)

    fold_filename = f'kcv_{dfname}_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_fold{args["fold"]:02}_{fold_filename}.pt'

    # instantiate objects
    torch.manual_seed(args["fold"])
    model = FullFVAE(**model_params)
    criterion = VAELoss(**loss_params)
    optimizer = optim.Adam(model.parameters(), **optim_params)

    # Adding the wandb watch statement ; Only add them in the script so that it never interferes anywhere in train_eval
    if args['log_wandb']:
        # wandb stuff
        wandb.init(project=unique_filename, name=f'fold_{args["fold"]:02}', config=args)
        wandb.watch(model, criterion=criterion, log_freq=len(train_loader))

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_metrics = train_eval_loops(args['n_epochs'], args['tolerance'], model,
                                                                   criterion,
                                                                   optimizer, train_dataset, train_loader, valid_loader,
                                                                   checkpoint_filename, outdir)

    # Convert list of dicts to dicts of lists
    train_losses_dict = {f'train_{key}': [d[key] for d in train_losses] for key in train_losses[0]}
    valid_losses_dict = {f'valid_{key}': [d[key] for d in valid_losses] for key in valid_losses[0]}
    train_metrics_dict = {f'train_{key}': [d[key] for d in train_metrics] for key in train_metrics[0]}
    valid_metrics_dict = {f'valid_{key}': [d[key] for d in valid_metrics] for key in valid_metrics[0]}

    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}

    # Saving text file for the run:
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Fold: {args["fold"]}')
        file.write(f"Best valid epoch: {best_epoch}\n")
        for k, v in best_val_loss.items():
            file.write(f'{k}:\t{v}\n')
        for k, v in best_val_metrics.items():
            file.write(f'{k}:\t{v}\n')

    pkl_dump(train_losses_dict, f'{outdir}/train_losses_{fold_filename}.pkl')
    pkl_dump(valid_losses_dict, f'{outdir}/valid_losses_{fold_filename}.pkl')
    pkl_dump(train_metrics_dict, f'{outdir}/train_metrics_{fold_filename}.pkl')
    pkl_dump(valid_metrics_dict, f'{outdir}/valid_metrics_{fold_filename}.pkl')
    plot_vae_loss_accs(losses_dict, accs_dict, unique_filename, outdir, dpi=300, palette='gnuplot2_r')

    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=checkpoint_filename,
                            dir_path=outdir)
    valid_preds = predict_model(model, valid_dataset, valid_loader)
    valid_preds['fold'] = args["fold"]
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{fold_filename}.csv', index=False)
    valid_seq_acc = valid_preds['seq_acc'].mean()
    valid_v_acc = valid_preds['v_correct'].mean()
    valid_j_acc = valid_preds['j_correct'].mean()
    print(f'Final valid mean accuracies (seq, v, j): \t{valid_seq_acc:.3%}\t{valid_v_acc:.3%}\t{valid_j_acc:.3%}')

    if args['test_file'] is not None:
        test_df = pd.read_csv(args['test_file'])
        test_basename = os.path.basename(args['test_file']).split(".")[0]
        test_dataset = CDR3BetaDataset(test_df, **dataset_params, v_map=train_dataset.v_map, j_map=train_dataset.j_map)
        test_loader = test_dataset.get_dataloader(batch_size=3 * args['batch_size'],
                                                  sampler=SequentialSampler)

        test_preds = predict_model(model, test_dataset, test_loader)
        test_preds['fold'] = args["fold"]
        test_preds.to_csv(f'{outdir}test_predictions_{test_basename}_{fold_filename}.csv', index=False)
        test_seq_acc = test_preds['seq_acc'].mean()
        test_v_acc = test_preds['v_correct'].mean()
        test_j_acc = test_preds['j_correct'].mean()
        print(f'Final valid mean accuracies (seq, v, j): \t{test_seq_acc:.3%}\t{test_v_acc:.3%}\t{test_j_acc:.3%}')
    else:
        test_seq_acc = None
        test_v_acc = None
        test_j_acc = None

    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:

        file.write(f'Fold: {kf}')
        file.write(f"Best valid seq acc: {valid_seq_acc}\n")
        file.write(f"Best valid V acc: {valid_v_acc}\n")
        file.write(f"Best valid J acc: {valid_j_acc}\n")
        if args['test_file'] is not None:
            file.write(f"Best test seq acc: {test_seq_acc}\n")
            file.write(f"Best test V acc: {test_v_acc}\n")
            file.write(f"Best test J acc: {test_j_acc}\n")
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
