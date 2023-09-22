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
from src.vautoencoders import *
from tqdm.auto import tqdm 
from torch.utils.data import BatchSampler, RandomSampler
from src.datasets import * 

def train_model(model, criterion, optimizer, epoch, train_loader, device):
    """Trains for ONE epoch and over all the batches"""
    model.train()
    criterion.train()
    train_loss = 0
    #Minibatch SGD, get a list of indices to separate the train data into batches 
    for data, target in tqdm(train_loader,
                  desc=f'Train Batch, epoch {epoch}:',
                  leave =False, position = 3):
        data = data.to(device)
        target = (x.to(device) for x in target)
        if model.__class__.__name__ == 'AutoEncoder':#AE
            pass
        #    _, x = model(data)
        #    loss = criterion(x.view(-1, onehot.shape[1], onehot.shape[2]), target)#Here was MSE loss
        elif model.__class__.__name__.startswith('VAE'):
            x_hat, mu, logvar = model(data)
            loss = criterion(x_hat, target, mu, logvar)

        #Optimizer
        model.zero_grad()
        loss.backward()
        optimizer.step()
        #average loss per sample within a batch
        train_loss += loss.item()
    train_loss /= math.ceil(train_loader)
    return train_loss


def eval_model(model, criterion, val_loader, device):
    model.eval()
    criterion.eval()
    val_loss= 0
    with torch.no_grad():
        for data,target in tqdm(val_loader,
                                desc = 'Valid batch',
                                leave =False, position = 4):
            data = data.to(device)
            target = (x.to(device) for x in target)
            if model.__class__.__name__ == 'AutoEncoder':#AE
                pass
            #    _, x = model(onehot)
            #    loss = criterion(x.view(-1, onehot.shape[1], onehot.shape[2]), onehot)
            
            elif model.__class__.__name__.startswith('VAE'):
                x_hat, mu, logvar = model(onehot)
                loss = criterion(x_hat, onehot, mu, logvar)

            val_loss += loss.item()

    val_loss /= math.ceil(len(val_loader)) # Divide by the number of batches
    
    return val_loss


def test_model(model, criterion, test_loader, device):
    """Computes the loss, and also reconstructs the 'predicted' sequence and compute metrics on it"""
    model.eval()
    criterion.eval()
    results_dict = {'accuracy':[],
                'precision':[],
                'recall':[],
                'f1':[],
                'hamming_total':[],
                'hamming_sequence':[]} 
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_loader,
                      desc = 'test set', leave =False):
            
            sequences = dataset[b] 
            onehot = onehot_batch(sequences, max_len = max_len, weighted = weighted,  pad = pad,
                                  positional = positional, atchley=atchley).to(device, dtype = torch.float32)
            
            if model.__class__.__name__.startswith('VAE'):
                decoded, _, _ = model(onehot)
            if model.__class__.__name__.startswith('AutoEncoder'):
                _, decoded = model(onehot)
                
            loss = criterion(decoded.view(-1, onehot.shape[1] * onehot.shape[2]), 
                             onehot.view(-1, onehot.shape[1] * onehot.shape[2]))
            losses.append(loss.item())
            #Reconstructing 
            decoded = decoded.view(-1, onehot.shape[1], onehot.shape[2])
            # Getting rid of the augmented positional vector
            # otherwise argmax will not have the intended behaviour for reconstruction
            if positional == True or atchley == True: 
                decoded = decoded[:,:,:21]
                onehot = onehot[:,:,:21]
                
            decoded = torch.argmax(decoded, dim = 2)
            decoded = F.one_hot(decoded).view(-1, onehot.shape[1]*onehot.shape[2]).cpu() # == 'y_pred'

            if weighted is not None:
                onehot[onehot==weighted] = 1
            onehot = onehot.to(dtype=torch.int64).view(-1, onehot.shape[1]*onehot.shape[2]).cpu() # == 'y_true'
            seq_original = decode_batch(onehot.view(-1, max_len, 21))
            seq_reconstructed = decode_batch(decoded.view(-1, max_len, 21))
            metrics = compute_metrics(onehot, decoded, seq_original, seq_reconstructed)
            for k in metrics.keys():
                results_dict[k].append(metrics[k])
    
    df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1','hamming_total','hamming_sequence', 'Criterion_loss'], index=['mean','var'])
    for k in results_dict.keys():
        df.loc['mean',k] = np.mean(results_dict[k])
        df.loc['var', k] = np.var(results_dict[k]) 
        
    df.loc['mean','Criterion_loss'] = np.mean(losses)
    df.loc['var','Criterion_loss'] = np.var(losses)
    
    df = df.rename(columns={'accuracy':'Accuracy', 
                        'precision':'Precision', 
                        'recall':'Recall',
                        'f1':'F1_score', 
                        'hamming_total':'Hamming_loss_with_pad',
                        'hamming_sequence':'Hamming_loss_sequence'})
    return df
            

    
def test_decode(model, values_array, n, max_len = 23, weighted = 1.6, pad = 'before', positional = False, tqdm_write=False, atchley = False):
    samples = np.random.choice(values_array, n)
    onehot = onehot_batch(samples, max_len = max_len, weighted = weighted, pad = pad, positional = positional,  atchley = atchley)
    #model.eval()
    with torch.no_grad():
        onehot = onehot.to('cuda')
        model.to('cuda')
        if model.__class__.__name__.startswith('VAE'):
            #print(model.training)
            xs_hat, _, _ = model(onehot)
            decoded = xs_hat[0] #First element of tuple
        elif model.__class__.__name__.startswith('AutoEncoder'):
            _, decoded = model(onehot)
    
    decoded = decoded.view(-1,onehot.shape[1], onehot.shape[2])
    
    if positional == True or atchley == True:
        decoded = decoded[:,:,:21]
        onehot = onehot[:,:,:21]
    decoded = torch.argmax(decoded, dim = 2)
    decoded = F.one_hot(decoded)
    decoded = decode_batch(decoded)
    
    original = decode_batch(onehot)
    if tqdm_write == True : 
        for x,y in zip(original,decoded):
            tqdm.write(f'\nreal:\t\t{x}\ndecoded:\t{y}')
    
    else: 
        for x,y in zip(original,decoded):
            print(f'\nreal:\t\t{x}\ndecoded:\t{y}')
    return original , decoded


def train_eval(model, criterion, optimizer, train_loader, val_loader, test_sequences, batchsize, 
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
            drop = int(adaptive[0])
            fraction = adaptive[1]
            #Every nb_epochs/drop the learning rate is divided
            epochs_drop = nb_epochs / drop
            for g in optimizer.param_groups:
                g['lr'] *= math.pow(fraction, math.floor((1+e)/epochs_drop))
        
        train_loss = train_model(model, criterion, optimizer, e, train_loader, device)
        train_losses.append(train_loss)
        val_loss = eval_model(model, criterion, val_loader, device)   
        
        tqdm.write(f'Losses at {e} epochs: \tTRAIN: {train_loss:.4e}\tVAL: {val_loss:.4e}')
        if e%1==0:
            _,_ = test_decode(model, np.random.choice(test_sequences ,3), 1, max_len=max_len, pad = pad, weighted=weighted, positional=positional, atchley = atchley, tqdm_write = True)    
            if not model.__class__.__name__.startswith('VAE'):
                tmp = test_model(model, criterion, dataset=np.random.choice(valid_dataset, math.floor(0.3*len(valid_dataset))), batchsize = batchsize, max_len = max_len, weighted = weighted, 
                                 pad = pad, device = device, positional = positional, atchley = atchley)
                display(tmp)
        
        if e != 0 and val_loss < best_val:
            best_val = val_loss
            torch.save({'state_dict':model.state_dict(), 'epoch':e}, os.path.join(outdir,filename+'.pth.tar'))
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

def tune_vae_vj(train_loader,
             val_loader,
             test_sequences,
             nb_epochs = 20,
             latent_dim = 70,
             batchsize = 2**15,
             lr = 5e-4,
             adaptive = (3, .96),
             wd = 1e-3,
             how = 'MSE',
             alpha = 1,
             beta = 2e-6, 
             cyclic=False,
             hyperbolic = False,
             gamma = None,
             v_weight = .75,
             j_weight = 0.1,
             weighted = 1.6,
             pad = 'after',
             act = nn.SELU(),
             positional=False,
             device = 'cuda',
             outdir = os.getcwd(),
             name = ''
    ):
    torch.cuda.empty_cache()
    criterion = VAELoss_vj(how, alpha, beta, v_weight, j_weight, 
                           cyclic, hyperbolic, gamma)
    if positional==True: aa_dim = 25
    else: aa_dim = 21
    name = '_'.join([f'VJ_VAE{name}-weighted{weighted}',
                     'latent'+str(latent_dim),
                     f'Pad-{pad}',f'cyclic{cyclic}-tanh{hyperbolic}-gamma{gamma}'])

    
    VAE_redo = VAE_VJ_tune(seq_len = 23, aa_dim = aa_dim, latent_dim = latent_dim, act=act)
    VAE_redo = VAE_redo.to(device)

    #optimizer = torch.optim.Adam(VAE_redo.parameters(), lr = lr)
    optimizer = torch.optim.AdamW(VAE_redo.parameters(), lr = lr, weight_decay=wd)
    print(f'\n\t-HOW: {how}\n\t-ALPHA: {alpha:.2e}\n\t-BETA: {beta:.2e}\n\t-adaptive: {adaptive}\n\t-wd: {wd:.2e}\n\t-weighted: {weighted}\n\t-positional: {positional}\n\t-act:{act}\n\t-latdim: {latent_dim}')
    train_losses, val_losses = train_eval(VAE_redo, criterion, optimizer, 
                                          train_loader, val_loader, test_sequences,
                                          batchsize, max_len= 23, weighted=weighted, 
                                          pad = pad,
                                          positional = positional, device=device, 
                                          lr= lr, nb_epochs = nb_epochs , outdir=outdir,
                                          filename = name, adaptive=adaptive)
    
    state_dict = torch.load(os.path.join(outdir, name+'.pth.tar'))['state_dict']
    VAE_redo.load(state_dict)
    #df = test_model(VAE_redo, nn.MSELoss(), test_dataset, batchsize, 23,
    #                weighted, pad, positional=positional, atchley=False, device=device)
    #display(df)
    df.to_csv(os.path.join(outdir, 'results_df.csv'), header=True, index=False)
    test_decode(VAE_redo, test_sequences, n=5, max_len = 23, weighted = weighted, pad = pad,
                positional = positional, tqdm_write = True)

