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
from torch import cuda
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists,  get_class_initcode_keys
from src.torch_utils import save_checkpoint, load_checkpoint, save_model_full, load_model_full
from src.models import FullTCRVAE, AttentionPeptideClassifier
from src.train_eval import predict_classifier, classifier_train_eval_loops
from src.datasets import LatentTCRpMHCDataset
from src.metrics import CombinedVAELoss, get_metrics
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
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
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

    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=0,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=0,
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
    parser.add_argument('-pepenc', '--pep_encoding', dest='pep_encoding', type=str, default='categorical',
                        help='Which encoding to use for the peptide (onehot, BL50LO, BL62LO, BL62FREQ, categorical; Default = categorical)')

    """
    Models args 
    """
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'If used, -pt_file and -json_file are not required and will attempt to read the .pt and .json from the provided directory')
    parser.add_argument('-pt_file', type=str, required=False,
                        default=None, help='Path to the checkpoint file to reload the VAE model')
    parser.add_argument('-json_file', type=str, required=False,
                        default=None, help='Path to the json file to reload the VAE model')

    # Classifier stuff
    parser.add_argument('-nh', '--n_hidden', dest='n_hidden_clf', type=int, default=32,
                        help='Number of hidden units in the Classifier. Default = 32')
    parser.add_argument('-do', dest='dropout', type=float, default=0,
                        help='Dropout percentage in the hidden layers (0. to disable)')
    parser.add_argument('-bn', dest='batchnorm', type=str2bool, default=False,
                        help='Use batchnorm (True/False)')
    parser.add_argument('-n_layers', dest='n_layers', type=int, default=0,
                        help='Number of hidden layers. Default is 0. (Architecture is in_layer -> [hidden_layers]*n_layers -> out_layer)')
    parser.add_argument('-num_heads', dest='num_heads', type=int, default=4,
                        help='Number of heads in the multihead attention. Default is 4')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=1e-4, required=False,
                        help='Learning rate for the optimizer. Default = 1e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-3, required=False,
                        help='Weight decay for the optimizer. Default = 1e-3')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=256, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=2000, required=False,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-4, required=False,
                        help='Tolerance for loss variation to log best model')
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
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()


def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    seed = args['seed'] if args['fold'] is None else args['fold']
    assert not all([args[k] is None for k in ['model_folder', 'pt_file', 'json_file']]), \
        'Please provide either the path to the folder containing the .pt and .json or paths to each file (.pt/.json) separately!'
    if args['model_folder'] is not None:
        try:
            checkpoint_file = next(
                filter(lambda x: x.startswith('checkpoint') and x.endswith('.pt'), os.listdir(args['model_folder'])))
            json_file = next(
                filter(lambda x: x.startswith('checkpoint') and x.endswith('.json'), os.listdir(args['model_folder'])))
            vae = load_model_full(args['model_folder'] + checkpoint_file, args['model_folder'] + json_file)
        except:
            print(args['model_folder'], os.listdir(args['model_folder']))
            raise ValueError(f'\n\n\nCouldn\'t load your files!! at {args["model_folder"]}\n\n\n')
    else:
        vae = load_model_full(args['pt_file'], args['json_file'])

    if torch.cuda.is_available() and args['cuda']:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print("Using : {}".format(device))
    torch.manual_seed(seed)

    # Convert the activation string codes to their nn counterparts
    df = pd.read_csv(args['file'])
    dfname = args['file'].split('/')[-1].split('.')[0]
    train_df = df.query('partition!=@args["fold"]')
    valid_df = df.query('partition==@args["fold"]')
    args['n_batches'] = math.ceil(len(train_df) / args['batch_size'])
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
    model_keys = get_class_initcode_keys(AttentionPeptideClassifier, args)
    dataset_keys = get_class_initcode_keys(LatentTCRpMHCDataset, args)

    model_params = {k: args[k] for k in model_keys}
    model_params['latent_dim'] = vae.latent_dim
    model_params['pep_dim'] = df.peptide.apply(len).max().item() if args['pep_encoding'] == 'categorical' else 12 * 20
    assert (model_params['latent_dim']+model_params['pep_dim'])%args['num_heads']==0, 'Wrong numheads!'
    dataset_params = {k: args[k] for k in dataset_keys}
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # Dumping args to file
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
    # Here, don't specify V and J map to use the default V/J maps loaded from src.data_processing
    train_dataset = LatentTCRpMHCDataset(vae, train_df, **dataset_params)
    valid_dataset = LatentTCRpMHCDataset(vae, valid_df, **dataset_params)
    # Random Sampler for Train; Sequential for Valid.
    # Larger batch size for validation because we have enough memory
    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args["batch_size"] * 2, sampler=SequentialSampler)

    fold_filename = f'kcv_{dfname}_f{args["fold"]:02}_{unique_filename}'
    checkpoint_filename = f'checkpoint_best_CLASSIFIER_fold{args["fold"]:02}_{fold_filename}.pt'

    # instantiate objects
    torch.manual_seed(args["fold"])
    model = AttentionPeptideClassifier(**model_params)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), **optim_params)

    # Adding the wandb watch statement ; Only add them in the script so that it never interferes anywhere in train_eval
    if args['log_wandb']:
        # wandb stuff
        wandb.init(project=unique_filename, name=f'fold_{args["fold"]:02}', config=args)
        wandb.watch(model, criterion=criterion, log_freq=len(train_loader))

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_metrics = classifier_train_eval_loops(args['n_epochs'], args['tolerance'],
                                                                              model,
                                                                              criterion, optimizer, train_loader,
                                                                              valid_loader,
                                                                              checkpoint_filename, outdir)

    # Convert list of dicts to dicts of lists
    train_losses_dict = {'train_loss': train_losses} #get_dict_of_lists(train_losses, 'train', filter=['auc', 'auc01'])
    train_metrics_dict = get_dict_of_lists(train_metrics, 'train', filter=['auc', 'auc01'])
    valid_losses_dict = {'valid_loss': valid_losses} #get_dict_of_lists(valid_losses, 'valid', filter=['auc', 'auc01'])
    valid_metrics_dict = get_dict_of_lists(valid_metrics, 'valid', filter=['auc', 'auc01'])

    losses_dict = {**train_losses_dict, **valid_losses_dict}
    accs_dict = {**train_metrics_dict, **valid_metrics_dict}

    # Saving text file for the run:
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write(f'Fold: {args["fold"]}\n')
        file.write(f"Best valid epoch: {best_epoch}\n")
        file.write(f'Best val loss: {best_val_loss}\n')
        for k, v in best_val_metrics.items():
            file.write(f'{k}:\t{v}\n')

    pkl_dump(train_losses_dict, f'{outdir}/train_losses_{fold_filename}.pkl')
    pkl_dump(valid_losses_dict, f'{outdir}/valid_losses_{fold_filename}.pkl')
    pkl_dump(train_metrics_dict, f'{outdir}/train_metrics_{fold_filename}.pkl')
    pkl_dump(valid_metrics_dict, f'{outdir}/valid_metrics_{fold_filename}.pkl')
    plot_vae_loss_accs(losses_dict, accs_dict, unique_filename, outdir,
                       dpi=300, palette='gnuplot2_r', warm_up=0)

    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=checkpoint_filename,
                            dir_path=outdir)
    valid_preds = predict_classifier(model, valid_dataset, valid_loader)
    valid_preds['fold'] = args["fold"]
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{fold_filename}.csv', index=False)
    print('Validation predictions: Per peptide performance')
    with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
        file.write('Validation preds ; Per peptide metrics\n')
        for pep in sorted(valid_preds.peptide.unique()): #valid_preds.groupby('peptide').agg(count=('B3', 'count')).sort_values('count', ascending=False).index:
            tmp = valid_preds.query('peptide==@pep')
            metrics = get_metrics(tmp['binder'].values, tmp['pred_prob'])
            print(f'{pep}:\tAUC: {metrics["auc"]:.4f}\tAUC_01: {metrics["auc_01"]:.4f}\tAP: {metrics["AP"]}')
            file.write(f'{pep}:\tAUC: {metrics["auc"]:.4f}\tAUC_01: {metrics["auc_01"]:.4f}\tAP: {metrics["AP"]}\n')

    if args['test_file'] is not None:
        test_df = pd.read_csv(args['test_file'])
        test_basename = os.path.basename(args['test_file']).split(".")[0]
        test_dataset = LatentTCRpMHCDataset(vae, test_df, **dataset_params)
        test_loader = test_dataset.get_dataloader(batch_size=3 * args['batch_size'],
                                                  sampler=SequentialSampler)

        test_preds = predict_classifier(model, test_dataset, test_loader)
        test_preds['fold'] = args["fold"]
        test_preds.to_csv(f'{outdir}test_predictions_{test_basename}_{fold_filename}.csv', index=False)

        with open(f'{outdir}args_{unique_filename}.txt', 'a') as file:
            file.write('Test preds ; Per peptide metrics\n')
            for pep in sorted(
                    test_preds.peptide.unique()):  # valid_preds.groupby('peptide').agg(count=('B3', 'count')).sort_values('count', ascending=False).index:
                tmp = test_preds.query('peptide==@pep')
                metrics = get_metrics(tmp['binder'].values, tmp['pred_prob'])
                print(f'{pep}:\tAUC: {metrics["auc"]:.4f}\tAUC_01: {metrics["auc_01"]:.4f}\tAP: {metrics["AP"]}')
                file.write(f'{pep}:\tAUC: {metrics["auc"]:.4f}\tAUC_01: {metrics["auc_01"]:.4f}\tAP: {metrics["AP"]}\n')

    best_dict = {'Best epoch': best_epoch}
    best_dict['val_loss'] = best_val_loss
    best_dict.update(best_val_metrics)
    # TODO : fix this
    # save_model_full(model, checkpoint_filename, outdir,
    #                 best_dict=best_dict, dict_kwargs=model_params)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
