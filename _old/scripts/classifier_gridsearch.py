from __future__ import print_function, division
#Allows relative imports
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#imports from files
from src.preprocessing import *
from src.VAE_train import *
from vae_cel.vae_cel import *
from vae_cel.vae_cel_train import *
from vae_cel.DeepRC_VAE import *
from src.embedding_visualisation import * 
from src.loss_metrics import *
from src.pickling import *
from src.datasets import *

from tqdm.auto import tqdm 

import pandas as pd 
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#checking gpu status
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using : {}".format(device))
else:
    device = torch.device('cpu')
    print("Using : {}".format(device))
    
#Plot and stuff
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')

torch.cuda.empty_cache()
# Ignore warnings)
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder

def train_model(model, X_train, y_train, batch_size, criterion, optimizer, device='cuda'):
    model.train()
    train_loss = 0
    for b in BatchSampler(RandomSampler(range(len(y_train))), batch_size = batch_size,
                               drop_last = False):
                     #position = 1,
                     #leave = False):
        values = X_train[b].to(device)
        target = y_train[b].to(device)
        score = model(values)
        t_loss = criterion(score, target)
        model.zero_grad()
        t_loss.backward()
        optimizer.step()
        train_loss += t_loss.item()   
        
    train_loss /= math.ceil((len(X_train)/batch_size))
    return train_loss

def val_model(model, X_val, y_val, batch_size, criterion, device='cuda'):
    model.eval()
    val_loss = 0
    for b in BatchSampler(RandomSampler(range(len(y_val))), batch_size = batch_size,
                           drop_last = False):
        
        values = X_val[b].to(device)
        target = y_val[b].to(device)
        with torch.no_grad():
            score = model(values)
        v_loss = criterion(score, target)
        val_loss += v_loss.item()
        
    val_loss /= math.ceil((len(X_val)/batch_size))
    return val_loss

def test_model(model, X_test, y_test, batch_size, device='cuda', binary = False):
    model.eval()
    rocs = []
    accs = []
    for b in BatchSampler(RandomSampler(range(len(y_test))), batch_size = batch_size,
                           drop_last = False):
        values = X_test[b].to(device)
        target = y_test[b].to(device)
        y_true = y_test[b].cpu().numpy()
        with torch.no_grad():
            if binary == False:
                score = F.softmax(model(values), dim =1).cpu()
                preds = torch.argmax(score,dim=1).numpy()
                if y_val.max() >1:
                    multi = 'ovo'
                    average = 'macro'
                else:
                    multi = 'ovr'
                    average = 'macro'
                    score = score[:,1]
                roc_auc = roc_auc_score(y_true, score.numpy(), average = average,
                            multi_class = multi)
            elif binary == True:
                score = model(values).cpu() #Score for positive class
                preds = quantize(score, threshold=0.5).numpy()
                roc_auc = roc_auc_score(y_true, score.numpy())
        
        
        acc = accuracy_score(y_true, preds)
        rocs.append(roc_auc)
        accs.append(acc)
        
    return np.mean(rocs), np.mean(accs)
    
def train_clf(model, data:tuple, loss_fct = nn.CrossEntropyLoss, 
              optimizer_module = torch.optim.AdamW,
              lr=5e-5, wd=1e-4, nb_epochs=1000, 
              batch_size=2**9, loss_weight = None, device='cuda'):
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    print(X_train.device, X_val.device, X_test.device, y_train.device, y_val.device, y_test.device)
    if loss_weight is not None:
        weights = loss_weight
    else:
        tmp = int(torch.cat((y_train,y_val), dim=0).max().item()+1)
        weights = None#torch.ones((tmp,))
        
    #weights = 1/(top5_nn.groupby('antigen_epitope').agg(count=('cdr3','count')).values/len(top5_nn))
    criterion = loss_fct(weight=None)
    optimizer = optimizer_module(model.parameters(), lr = lr, weight_decay= wd)
    
    binary = "BCE" in (criterion.__class__.__name__)
    print('BINARY', binary)

    train_losses = []
    val_losses = []
    rocs = []
    accs = []
    broken = False
    model = model.to(device)
    for e in tqdm(range(nb_epochs),
                 position = 0, leave = False):
        train_loss = train_model(model, X_train, y_train, batch_size, criterion, optimizer, device)
        val_loss = val_model(model, X_val, y_val, batch_size, criterion, device)
        #train
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss > 1.2 and e > 200:
            broken = True
            break
        if e%25 == 0 or e==nb_epochs-1:
            roc_auc, acc = test_model(model, X_test, y_test, batch_size, device, binary)
            print(f'Epoch:{e};\tTrain: {train_loss:.3e}\tVal: {val_loss:.3e}'\
                  f'\n\t\tROC AUC :{roc_auc:.3f}, \taccuracy: {acc:.3f}')
            rocs.append(roc_auc)
            accs.append(acc)
    losses = {'train': train_losses,
              'val' : val_losses,
              'roc': rocs,
              'acc': accs}
    cs = ['b-', 'r-']#, 'g-.', 'm-.']
    
    for k, c in zip(losses.keys(), cs):
        plt.plot(losses[k], color = c[0], ls = c[1], label = k)
    
    if broken == False:
        x = np.arange(start=0,stop=nb_epochs,step=25)
        x = np.append(x, nb_epochs-1)
        plt.plot(x, rocs, color = 'g', ls = '-.', marker = 'o', markersize = 5, label = 'ROC AUC')
        plt.plot(x, accs, color = 'm', ls = '-.', marker = 'x', markersize = 5, label = 'Accuracy')
        
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    return losses

class MLP_binary(nn.Module):
    def __init__(self, n_layers, n_hidden, activation = nn.SELU(), p_drop = 0.5):
        super(MLP_binary, self).__init__()
        if p_drop >0 and p_drop <1:
            self.drop = nn.Dropout(p_drop)
        else: 
            self.drop = nn.Identity()
            
        self.input_layers = nn.Sequential(nn.Linear(100, 256),
                                          nn.BatchNorm1d(256),
                                          activation,
                                          self.drop,
                                          nn.Linear(256, 512),
                                          nn.BatchNorm1d(512),
                                          activation,
                                          self.drop,
                                          nn.Linear(512, n_hidden),
                                          activation,
                                          self.drop)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation)
            layers.append(self.drop) 
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_hidden//2),
                                    nn.BatchNorm1d(n_hidden//2),
                                    activation,
                                    self.drop,
                                    
                                    nn.Linear(n_hidden//2, 10),
                                    activation,
                                    
                                    nn.Linear(10, 1))

    def forward(self, x):
        x = self.input_layers(x)
        x = self.hidden_layers(x)
        x = self.output(x) #No activation because I want to return logits for the BCELoss
        return x.view(-1,)


def get_min(array, x_epochs):
    index = np.argmin(array)
    value = array[index]
    epoch = x_epochs[index]
    return epoch, value

def get_max(array, x_epochs):
    index = np.argmax(array)
    value = array[index]
    epoch = x_epochs[index]
    return epoch, value


def main():
  print('reading csv')
  subset = pd.read_csv('../training_data_new/db/subset_labeled_embedded.csv')
  
  torch.manual_seed(20)
  X = torch.tensor(subset[['z_'+str(i) for i in range(100)]].values)
  y = torch.tensor(subset['cd_label'].values).long() # Doing cd4 vs cd8 binary classification
  print('splitting data')
  # Split X into Train/Test
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state = 20)
  # Split Train into Train/Val
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, random_state = 20)

  X_train = X_train.float()#.to('cuda')
  X_val = X_val.float()#.to('cuda')
  X_test = X_test.float()#.to('cuda')
  y_train = y_train.float()#.to('cuda')
  y_val = y_val.float()#.to('cuda')
  y_test = y_test.float()#.to('cuda')
  

  nb_epochs = 400
  x_epochs = np.arange(0, nb_epochs, 25)
  x_epochs = np.append(x_epochs, nb_epochs-1)
  loss_dict = {};
  results = pd.DataFrame(columns = ['params', 'best_roc','roc_epoch', 'best_val','val_epoch'])
  print('starting training')
  for n_layers in tqdm([2,3,4], 
                       leave = False,
                       position = 0):
      for n_hidden in tqdm([64,128,256], 
                           leave = False,
                           position = 1):
          for lr in tqdm([1e-2, 1e-3, 1e-4], 
                         leave = False,
                         position = 2):
              for wd in tqdm([1e-2, 1e-5, 1e-8], 
                             leave = False,
                             position = 3):
                  params = '_'.join([f'n_layers{n_layers}',
                                    f'hidden{n_hidden}', 
                                    f'lr{lr:.1e}',
                                    f'wd{wd}'])
                  print(f"n_layers{n_layers}\tn_hidden{n_hidden}\tlr{lr:.1e}\twd{wd:.1e}")
                  model =MLP_binary(n_layers=n_layers, n_hidden=n_hidden,
                                    activation = nn.SELU(), p_drop = 0.4).to(dtype=torch.float32)
                  
                  losses = train_clf(model, (X_train, X_val, X_test, y_train, y_val, y_test),
                                     loss_fct = nn.BCEWithLogitsLoss, lr = lr,
                                     wd = wd, nb_epochs = nb_epochs, batch_size = 2**14,
                                     loss_weight = None, device = 'cuda')
                  loss_dict[params] = losses 
                  plt.savefig(f'../output/classifier_gridsearch/{params}.jpg')
                  plt.close()
                  roc_epoch, best_roc = get_max(losses['roc'], x_epochs)
                  val_epoch, best_val = get_min(losses['val'], range(nb_epochs))
                  results = results.append(pd.DataFrame(data=[[params, best_roc, roc_epoch, best_val, val_epoch]],
                                              columns = ['params','best_roc','roc_epoch','best_val','val_epoch']),
                                 ignore_index=True)
                  del model
                  torch.cuda.empty_cache()

              results.to_csv('../output/classifier_gridsearch/gridsearch_output.csv', header=True, index = False)
              time.sleep(6) 
  save_pkl('../output/classifier_gridsearch/losses.pkl', loss_dict)

if __name__ == '__main__':
  main()
