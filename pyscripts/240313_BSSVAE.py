import pandas as pd
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import wandb
import torch
from torch import optim
from torch import cuda
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists, get_class_initcode_keys
from src.torch_utils import load_checkpoint, save_model_full, load_model_full, get_available_device
from src.multimodal_models import BSSVAE
from src.multimodal_train_eval import predict_multimodal, multimodal_train_eval_loops
from src.datasets import MultimodalPepTCRDataset
from src.metrics import BSSVAELoss
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-logwb', '--log_wandb', dest='log_wandb', required=False, default=False,
                        type=str2bool, help='Whether to log a run using WandB. False by default')
    parser.add_argument('-f', '--file', dest='file', required=False, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
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
    parser.add_argument('-mlpep', '--max_len_pep', dest='max_len_pep', type=int, default=12,
                        help='Max seq length admitted for peptide. Set to 0 to disable adding peptide to the input')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool, default=False,
                        help='Adding positional encoding to the sequence vector. False by default')
    """
    Models args 
    """
    parser.add_argument('-nhtcr', '--hidden_dim_tcr', dest='hidden_dim_tcr', type=int, default=200,
                        help='Number of hidden units in the VAE. Default = 200')
    parser.add_argument('-nhpep', '--hidden_dim_pep', dest='hidden_dim_pep', type=int, default=200,
                        help='Number of hidden units in the VAE. Default = 200')
    parser.add_argument('-nl', '--latent_dim', dest='latent_dim', type=int, default=100,
                        help='Size of the latent dimension. Default = 100')
    parser.add_argument('-act', '--activation', dest='activation', type=str, default='selu',
                        help='Which activation to use. Will map the correct nn.Module for the following keys:' \
                             '[selu, relu, leakyrelu, elu]')
    # Not implemented as of now
    parser.add_argument('-do', dest='dropout', type=float, default=0.25,
                        help='Dropout percentage in the hidden layers (0. to disable)')
    parser.add_argument('-bn', dest='batchnorm', type=str2bool, default=True,
                        help='Use batchnorm (True/False)')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=5e-4, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=512, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=5000, required=False,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=1,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld_n', '--weight_kld_n', dest='weight_kld_n', type=float, default=1e-2,
                        help='Which weight to use for the KLD (normal) term in the loss')
    parser.add_argument('-lwkld_z', '--weight_kld_z', dest='weight_kld_z', type=float, default=1,
                        help='Which weight to use for the KLD (Latent) term in the loss')
    # parser.add_argument('-lwtrp', '--weight_triplet',
    #                     dest='weight_triplet', type=float, default=1, help='Weight for the triplet loss term')
    # parser.add_argument('-dist_type', '--dist_type', dest='dist_type', default='cosine', type=str,
    #                     help='Which distance metric to use [cosine, l2, l1]')
    # parser.add_argument('-margin', dest='margin', default=None, type=float,
    #                     help='Margin for the triplet loss (Default is None and will have the default behaviour depending on the distance type)')
    parser.add_argument('-wukld', '--warm_up_kld', dest='warm_up_kld', type=int, default=10,
                        help='Whether to do a warm-up period for the loss (without the KLD term). ' \
                             'Default = 10. Set to 0 if you want this disabled')
    parser.add_argument('-kldts', '--kld_tahn_scale', dest='kld_tahn_scale', type=float, default=0.1,
                        help='Scale for the TanH annealing in the KLD_n term')
    parser.add_argument('-fp', '--flat_phase', dest='flat_phase', default=None, type=int,
                        help='If used, the duration (in epochs) of the "flat phase" in the KLD annealing')

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


def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    seed = args['seed'] if args['fold'] is None else args['fold']

    # Redundant usage because clf and vae use a diff variable name (was useful before should probly be phased out now)
    args['n_latent'] = args['latent_dim']
    if torch.cuda.is_available() and args['cuda']:
        device = get_available_device()
    else:
        device = torch.device('cpu')
    print("Using : {}".format(device))
    torch.manual_seed(seed)
    # Convert the activation string codes to their nn counterparts
    args['activation'] = {'selu': nn.SELU(), 'relu': nn.ReLU(),
                          'leakyrelu': nn.LeakyReLU(),
                          'elu': nn.ELU()}[args['activation']]
    # Loading data and getting train/valid
    # TODO: Restore valid kcv behaviour // or not
    df = pd.read_csv(args['file'])
    dfname = args['file'].split('/')[-1].split('.')[0]
    if args['debug']:
        df = df.sample(frac=0.15)

    fold = args["fold"]
    train_df = df.query('partition!=@fold')
    valid_df = df.query('partition==@fold')
    # TODO: get rid of this bad hardcoded behaviour for AA_dim ; Let's see if we end up using Xs
    args['aa_dim'] = 20
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
    model_keys = get_class_initcode_keys(BSSVAE, args)
    dataset_keys = get_class_initcode_keys(MultimodalPepTCRDataset, args)
    loss_keys = get_class_initcode_keys(BSSVAELoss, args)
    # Get params from args
    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    loss_params = {k: args[k] for k in loss_keys}
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
    # Here, don't specify V and J map to use the default V/J maps loaded from src.data_processing
    train_dataset = MultimodalPepTCRDataset(train_df, **dataset_params)
    valid_dataset = MultimodalPepTCRDataset(valid_df, **dataset_params)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory
    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)

    fold_filename = f'kcv_{dfname}_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_fold{args["fold"]:02}_{fold_filename}.pt'

    # instantiate objects
    torch.manual_seed(args["fold"])
    model = BSSVAE(**model_params)
    model.to(device)
    criterion = BSSVAELoss(**loss_params)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), **optim_params)

    # Adding the wandb watch statement ; Only add them in the script so that it never interferes anywhere in train_eval
    if args['log_wandb']:
        # wandb stuff
        wandb.init(project=unique_filename, name=f'fold_{args["fold"]:02}', config=args)
        wandb.watch(model, criterion=criterion, log_freq=len(train_loader))

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_metrics = multimodal_train_eval_loops(args['n_epochs'], model,
                                                                               criterion, optimizer, train_loader,
                                                                               valid_loader, checkpoint_filename, outdir,
                                                                               args['tolerance'])

    # Convert list of dicts to dicts of lists
    train_losses_dict = get_dict_of_lists(train_losses,
                                          'train')
    train_metrics_dict = get_dict_of_lists(train_metrics,
                                           'train')
    valid_losses_dict = get_dict_of_lists(valid_losses,
                                          'valid')
    valid_metrics_dict = get_dict_of_lists(valid_metrics,
                                           'valid')

    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}
    keys_to_remove = []

    for k in accs_dict:
        if 'precision' in k or 'auc_01' in k or k == 'train_accuracy' or k == 'valid_accuracy':
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del accs_dict[k]
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
                       dpi=300, palette='gnuplot2_r', warm_up=args['warm_up_kld'])

    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=checkpoint_filename,
                            dir_path=outdir)
    valid_preds = predict_multimodal(model, valid_dataset, valid_loader.batch_size)
    valid_preds['fold'] = args["fold"]
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{fold_filename}.csv', index=False)
    valid_seq_acc = valid_preds['seq_acc'].mean()

    print(f'Final valid reconstruction accuracy: \t{valid_seq_acc:.3%}')

    if args['test_file'] is not None:
        test_df = pd.read_csv(args['test_file'])
        test_basename = os.path.basename(args['test_file']).split(".")[0]
        test_dataset = MultimodalPepTCRDataset(test_df, **dataset_params)
        test_loader = test_dataset.get_dataloader(batch_size=3 * args['batch_size'],
                                                  sampler=SequentialSampler)

        test_preds = predict_multimodal(model, test_dataset, test_loader.batch_size)
        test_preds['fold'] = args["fold"]
        test_preds.to_csv(f'{outdir}test_predictions_{test_basename}_{fold_filename}.csv', index=False)
        test_seq_acc = test_preds['seq_acc'].mean()

        print(f'Final test reconstruction accuracy: \t{test_seq_acc:.3%}')
    else:
        test_seq_acc = None

    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:

        file.write(f'Fold: {kf}')
        file.write(f"Best valid seq acc: {valid_seq_acc}\n")
        if args['test_file'] is not None:
            file.write(f"Best test seq acc: {test_seq_acc}\n")
    best_dict = {'Best epoch': best_epoch}
    best_dict.update(best_val_loss)
    best_dict.update(best_val_metrics)
    # reshape dict for saving
    save_model_full(model, checkpoint_filename, outdir,
                    best_dict=best_dict, dict_kwargs=model_params)
    load_model_full(checkpoint_filename, f'{checkpoint_filename.split(".pt")[-2]}_JSON_kwargs.json',
                    outdir)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
