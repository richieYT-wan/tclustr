import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.nn.functional as F

#Plot and stuff
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')

from src.preprocessing import * 
from src.loss_metrics import *
from src.pickling import * 
from src.torch_util import *
from vae_cel.vae_cel import *
from vae_cel.vae_cel_loss import *

from tqdm.auto import tqdm 
from torch.utils.data import BatchSampler, RandomSampler


# should be like this : dataset = dataframe[['amino_acid','v_family','j_family']].values

def train_model(model, criterion, optimizer,epoch, dataset, 
                batch_size, max_len, weighted,
                pad, positional, atchley, device):
    """Trains for ONE epoch and over all the batches"""
    model.train()
    criterion.train()
    train_loss = 0
    #Minibatch SGD, get a list of indices to separate the train data into batches 
    for b in tqdm(BatchSampler(RandomSampler(dataset),
                          batch_size = batch_size, drop_last=False),
                  desc=f'Train Batch, epoch {epoch}:',
                  leave =False, position = 3):
        
        #Sending everything to device, x is a tuple of onehot-vectors (sequence_oh, v_oh, j_oh)
        x = batch_aa_vj(dataset[b,:], max_len, weighted, pad, positional, atchley, device)
        x_hat, mu, logvar = model(x) #x_hat is also a tuple
        loss = criterion(x_hat, x, mu, logvar)

        #Optimizer
        model.zero_grad()
        loss.backward()
        optimizer.step()
        #average loss per sample within a batch
        train_loss += loss.item()
    train_loss /= math.ceil((len(dataset)/batch_size))
    return train_loss


def eval_model(model, criterion, dataset, batch_size, max_len,
               weighted, pad, positional, atchley, device):
    model.eval()
    criterion.eval()
    val_loss= 0
    with torch.no_grad():
        for b in tqdm(BatchSampler(RandomSampler(dataset),batch_size = batch_size, 
                                   drop_last=False),
                      desc = 'Valid batch',
                      leave =False, position = 4):
            #Standard train loops
            x = batch_aa_vj(dataset[b,:], max_len, weighted, pad, positional, atchley, device)
            x_hat, mu, logvar = model(x)
            loss = criterion(x_hat, x, mu, logvar)
            val_loss += loss.item()

    val_loss /= math.ceil(len(dataset)/batch_size) # Divide by the number of batches
    
    return val_loss


def test_model(model, criterion, dataset, batch_size, max_len, weighted, pad, positional, atchley, device):
    """Computes the loss, and also reconstructs the 'predicted' sequence and compute metrics on it"""
    model.eval()
    if criterion.__class__.__name__.startswith('VAE'):
        criterion.eval()
    
    results_dict = {'hamming_seq_padded':[],
                    'hamming_sequence':[],
                    #'accuracy_seq':[],
                    'accuracy_V':[],
                    'accuracy_J':[]
                    } 
    losses = []
    with torch.no_grad():
        for b in tqdm(BatchSampler(RandomSampler(dataset),batch_size = batch_size, drop_last=False),
                      desc = 'test set', leave =False):
            #xs_Z are tuples!!!
            xs_true = batch_aa_vj(dataset[b,:], max_len, weighted, pad, positional, atchley, device)
            xs_hat, mu, logvar = model(xs_true) #x_hat is also a tuple
            loss = criterion(xs_hat, xs_true, mu, logvar)
            losses.append(loss.item())
            #Reconstructing 
            x_true, x_hat, seq_original, seq_reconstructed = reconstruct_seq_vj(xs_true, xs_hat, max_len, weighted, positional, atchley)
            metrics = compute_metrics(x_true, x_hat, seq_original, seq_reconstructed)
            for k in metrics.keys():
                results_dict[k].append(metrics[k])
    
    df = pd.DataFrame(columns = results_dict.keys(), index=['mean','var'])
    for k in results_dict.keys():
        df.loc['mean',k] = np.mean(results_dict[k])
        df.loc['var', k] = np.var(results_dict[k]) 
        
    df.loc['mean','Criterion_loss'] = np.mean(losses)
    df.loc['var','Criterion_loss'] = np.var(losses)
    
    return df
            

    
def test_decode(model, dataset, n, max_len = 23, weighted = 1.6, pad = 'before', positional = False, tqdm_write=False, atchley = False, device='cuda'):
    
    indices = np.random.choice(range(len(dataset)), n)
    samples = dataset[indices, :]
    
    model.eval()
    with torch.no_grad():
        model.to('cuda')
        (onehot, true_v, true_j) = batch_aa_vj(samples, max_len, weighted, pad, 
                                               positional, atchley, device) 
        (x_aa, x_v, x_j), _, _ = model((onehot, true_v, true_j))
        
    x_aa = x_aa.view(-1, onehot.shape[1], onehot.shape[2])
    
    if positional == True or atchley == True:
        x_aa = x_aa[:,:,:21]
        onehot = onehot[:,:,:21]
    true_v = list(torch.argmax(true_v, dim = 1).detach().cpu().numpy())
    true_j = list(torch.argmax(true_j, dim = 1).detach().cpu().numpy())
    x_v = list(torch.argmax(F.softmax(x_v, dim = 1), dim = 1).detach().cpu().numpy())
    x_j = list(torch.argmax(F.softmax(x_j, dim = 1), dim = 1).detach().cpu().numpy())
    decoded = torch.argmax(x_aa, dim = 2)
    decoded = F.one_hot(decoded)
    decoded = decode_batch(decoded)
    original = decode_batch(onehot)
    if tqdm_write == True : 
        for x,y, tv, xv, tj, xj in zip(original,decoded,true_v,x_v, true_j, x_j):
            tqdm.write(f'\nreal:\t\t{x} ;\tV:{tv}\tJ:{tj}'\
                       f'\ndecoded:\t{y} ;\tV:{xv}\tJ:{xj}\n')
    
    else: 
        for x,y, tv, xv, tj, xj in zip(original,decoded,true_v,x_v, true_j, x_j):
            print(f'\nreal:\t\t{x} ;\tV:{tv}\tJ:{tj}'\
                  f'\ndecoded:\t{y} ;\tV:{xv}\tJ:{xj}')
    return original , decoded


def train_eval(model, criterion, optimizer, train_dataset, valid_dataset, batch_size, 
               max_len, weighted, pad, positional, device, lr, nb_epochs, outdir, 
               filename='', adaptive=None, atchley = False):
    """
    The parameter ADAPTIVE is a very crude manual implementation of a lr scheduler.
    If used, it should be a tuple of the format (drop, fraction) 
    (see adaptive loop in training )
    
    With the following formula : 
    lr = lr0 * fraction^floor(epoch / drop)
    lr0 : initial learning rate
    fraction : how much we want to drop
    drop : after how many epochs we want to reduce the learning rate
    epoch : epoch number
    """
    train_losses = []
    val_losses = []
    
    #name, latdim, act, p_drop, lr = filename.split('_')
    tqdm.write(f'\nFor model : {filename}')
    
    best_val = 1e6 # Very large first initial loss value to check for best val and update
    for e in tqdm(range(nb_epochs), leave=False):
        sum_loss = 0
        # crude lr scheduler from scratch
        
        if adaptive:
            lr_0 = optimizer.param_groups[0]['lr']
            drop = int(adaptive[0])
            fraction = adaptive[1]
            #Once epochs_drop is reached, LR starts decreasing
            epochs_drop = math.ceil(nb_epochs / drop)
            for g in optimizer.param_groups:
                g['lr'] *= math.pow(fraction, math.floor((1+e)/epochs_drop))
                if g['lr'] < 0.1*lr_0:
                    adaptive = 'switch'
        
        if adaptive == 'switch': 
            tqdm.write('0.1 * learning rate reached.')
            for g in optimizer.param_groups:
                g['lr'] = 0.1*lr_0
            adaptive = None
                    
        train_loss = train_model(model, criterion, optimizer, e, train_dataset,
                                 batch_size, max_len, weighted, pad = pad, positional = positional, 
                                 device=device, atchley = atchley)
        train_losses.append(train_loss)
        val_loss = eval_model(model, criterion, valid_dataset,
                                 batch_size, max_len, weighted, pad = pad, positional = positional, 
                              device=device, atchley = atchley)   
        
        tqdm.write(f'Losses at {e} epochs: \tTRAIN: {train_loss:.4e}\tVAL: {val_loss:.4e}')
        if e%1==0:
            _,_ = test_decode(model, valid_dataset, 1, max_len=max_len, pad = pad, weighted=weighted, positional=positional, atchley = atchley, tqdm_write = True, device = device)    
        
        torch.save({'state_dict':model.state_dict(), 'epoch':e}, os.path.join(outdir,f'epoch{e}'+filename+'.pth.tar'))
        if e != 0 and val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict':model.state_dict(), 'epoch':e}, os.path.join(outdir,'BEST_'+filename+'.pth.tar'))
        val_losses.append(val_loss)

    f,a = plt.subplots(1,1,figsize=(10,7))
    plt.plot(train_losses, 'b-', label = 'train')
    plt.plot(val_losses, 'r-', label = 'val')
    a.set_title(filename)
    a.set_xlim([0, nb_epochs])
    a.set_xlabel('Epochs')
    a.set_ylabel('Loss')
    #a.set_ylim([1e-6, 1e-2])
    #plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(outdir,filename+'_losses.jpg'))
    
    return train_losses, val_losses

def tune_vae(train_dataset,
             valid_dataset,
             test_dataset,
             nb_epochs = 20,
             latent_dim = 70,
             batch_size = 2**15,
             lr = 5e-4,
             adaptive = (3, .96),
             wd = 1e-3,
             how = 'CEL',
             alpha = 1,
             beta = 2e-6, 
             cyclic=False,
             hyperbolic = False,
             linear = False,
             gamma = None,
             weighted = 1.6,
             pad = 'after',
             act = nn.SELU(),
             positional=False,
             device = 'cuda',
             outdir = os.getcwd(),
             name = ''
    ):
    torch.cuda.empty_cache()
    criterion = VAELoss_cel(alpha, beta, cyclic, hyperbolic, linear, gamma)
    if positional==True: aa_dim = 25
    else: aa_dim = 21
    
    anneal = 'None'
    if cyclic == True : anneal = 'cyclic'
    elif hyperbolic == True: anneal = 'hyper'
    elif linear == True : anneal = 'linear'
    name = '_'.join([f'{name}-weighted{weighted}',
                     'latent'+str(latent_dim),
                     f'Pad-{pad}',f'Annealing-{anneal}-gamma{gamma*batch_size/len(train_dataset)}'])
    
    model= VAE_cel(seq_len = 23, aa_dim = aa_dim, latent_dim = latent_dim, act=act,
                       v_dim = 30, j_dim = 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=wd)

    #Calling the function that does everything (train&eval)
    train_losses, val_losses = train_eval(model, criterion, optimizer, 
                                          train_dataset, valid_dataset,
                                          batch_size, max_len= 23, weighted=weighted, 
                                          pad = pad,
                                          positional = positional, device=device, 
                                          lr= lr, nb_epochs = nb_epochs , outdir=outdir,
                                          filename = name, adaptive=adaptive)
    losses = {'train':train_losses,
              'val':val_losses}
    save_pkl(os.path.join(outdir, f'{name}_losses.pkl'),losses)
    
    print('\n\t\t====== Reloading & Evaluating on test set ======\n')
    filename = [x for x in os.listdir(outdir) if ('BEST_' in x and name in x) and x.endswith('.pth.tar')][0]
    state_dict = torch.load(os.path.join(outdir,filename))['state_dict']
    model.load_state_dict(state_dict)
    df = test_model(model, criterion, test_dataset, batch_size, 23,
                    weighted, pad, positional=positional, atchley=False, device=device)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    df.to_csv(os.path.join(outdir, f'{name}_results_df.csv'), header=True, index=False)
    
    test_decode(model, test_dataset, n=5, max_len = 23, weighted = weighted,
                pad = pad, positional = positional, tqdm_write = True)
    

def resume_training(
                    train_dataset,
                    valid_dataset,
                    test_dataset,
                    weightpath=None,
                    nb_epochs = 20,
                    latent_dim = 70,
                    batch_size = 2**15,
                    lr = 5e-4,
                    adaptive = (3, .96),
                    wd = 1e-3,
                    how = 'CEL',
                    alpha = 1,
                    beta = 2e-6, 
                    cyclic=False,
                    hyperbolic = False,
                    linear = False,
                    gamma = None,
                    weighted = 1.6,
                    pad = 'after',
                    act = nn.SELU(),
                    positional=False,
                    device = 'cuda',
                    outdir = os.getcwd(),
                    name = ''
    ):
    torch.cuda.empty_cache()
    criterion = VAELoss_cel(alpha, beta, cyclic, hyperbolic, linear, gamma)
    if positional==True: aa_dim = 25
    else: aa_dim = 21
    
    anneal = 'None'
    if cyclic == True : anneal = 'cyclic'
    elif hyperbolic == True: anneal = 'hyper'
    elif linear == True : anneal = 'linear'
    name = '_'.join([f'{name}-weighted{weighted}',
                     'latent'+str(latent_dim),
                     f'Pad-{pad}',f'Annealing-{anneal}-gamma{gamma*batch_size/len(train_dataset)}'])
    
    model= VAE_cel(seq_len = 23, aa_dim = aa_dim, latent_dim = latent_dim, act=act,
                       v_dim = 30, j_dim = 2)
    model = load_model(model, weightpath)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=wd)

    #Calling the function that does everything (train&eval)
    train_losses, val_losses = train_eval(model, criterion, optimizer, 
                                          train_dataset, valid_dataset,
                                          batch_size, max_len= 23, weighted=weighted, 
                                          pad = pad,
                                          positional = positional, device=device, 
                                          lr= lr, nb_epochs = nb_epochs , outdir=outdir,
                                          filename = name, adaptive=adaptive)
    losses = {'train':train_losses,
              'val':val_losses}
    save_pkl(os.path.join(outdir, f'{name}_losses.pkl'),losses)
    
    print('\n\t\t====== Reloading & Evaluating on test set ======\n')
    filename = [x for x in os.listdir(outdir) if ('BEST_' in x and name in x) and x.endswith('.pth.tar')][0]
    state_dict = torch.load(os.path.join(outdir,filename))['state_dict']
    model.load_state_dict(state_dict)
    df = test_model(model, criterion, test_dataset, batch_size, 23,
                    weighted, pad, positional=positional, atchley=False, device=device)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    df.to_csv(os.path.join(outdir, f'{name}_results_df.csv'), header=True, index=False)
    
    test_decode(model, test_dataset, n=5, max_len = 23, weighted = weighted,
                pad = pad, positional = positional, tqdm_write = True)