import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
from tqdm.auto import tqdm
import os 

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 200
sns.set_style('darkgrid')

from torch.utils.data import BatchSampler, RandomSampler
from src.preprocessing import * 
from src.torch_util import * 
from src.embedding_visualisation import * 

import sklearn 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, roc_curve
from sklearn.preprocessing import LabelEncoder

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
    
def train_clf(model, data:tuple, loss_fct = nn.BCEWithLogitsLoss, 
              optimizer_module = torch.optim.AdamW,
              lr=5e-5, wd=1e-4, nb_epochs=1000, 
              batch_size=2**9, loss_weight = None, device='cuda', 
              outdir='', filename = ''):
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    if loss_weight is not None:
        weights = loss_weight
    else:
        weights = None#torch.ones((tmp,))
        
    #weights = 1/(top5_nn.groupby('antigen_epitope').agg(count=('cdr3','count')).values/len(top5_nn))
    if loss_fct.__name__ == 'BCEWithLogitsLoss':
        criterion = loss_fct(pos_weight=weights)
    else:
        criterion = loss_fct(weight=weights)
    optimizer = optimizer_module(model.parameters(), lr = lr, weight_decay= wd)
    
    binary = "BCE" in (criterion.__class__.__name__)
    train_losses = []
    val_losses = []
    rocs = []
    accs = []
    broken = False
    model = model.to(device)
    best_val = 100
    best_roc = 0 
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

        if outdir != '':
            if rocs[-1] > best_roc :
                best_roc = rocs[-1]
                torch.save({'state_dict':model.state_dict(), 'epoch':e,'roc_auc':best_roc}, 
                             os.path.join(outdir,f'best{filename}.pth.tar'))

    losses = {'train': train_losses,
              'val' : val_losses,
              'roc': rocs,
              'acc': accs}
    bests = {}
    cs = ['b-', 'r-']#, 'g-.', 'm-.']
    
    for k, c in zip(losses.keys(), cs):
        plt.plot(losses[k], color = c[0], ls = c[1], label = k)
    
    if broken == False:
        x = np.arange(start=0,stop=nb_epochs,step=25)
        x = np.append(x, nb_epochs-1)
        plt.plot(x, rocs, color = 'g', ls = '-.', marker = 'o', markersize = 2, label = 'ROC AUC')
        plt.plot(x, accs, color = 'm', ls = '-.', marker = 'x', markersize = 2, label = 'Accuracy')
        roc_epoch, best_roc = get_max(losses['roc'], x)
        val_epoch, best_val = get_min(losses['val'], range(nb_epochs))

        bests['roc'] = (roc_epoch, best_roc)
        bests['val'] = (val_epoch, best_val)
        
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(filename+f'\nBest ROC_AUC : {bests["roc"][1]:.3e} at {bests["roc"][0]} epochs')
    if outdir != '':
        plt.savefig(os.path.join(outdir, filename+'.png'), dpi=200)

    return losses, bests

def reload_predict(model, data:tuple, lr=5e-5, wd=1e-4, nb_epochs=1000, 
              batch_size=2**16, loss_weight = None, device='cuda', 
              outdir='', filename = ''):
    """I'm keeping the kwargs although they are not used because of how I defined kfold cv xd"""
    X_train, X_val, X_test, y_train, y_val, y_test = data
    if outdir != '':
        path_to_weight = os.path.join(outdir,f'best{filename}.pth.tar')
        model = load_model(model, path_to_weight)
    model.eval()
    model.to(device)
    X = X_test.to(device)
    y = y_test.to(device)
    with torch.no_grad():
        pred_scores = F.sigmoid(model(X)).cpu()
    curve = roc_curve(y_test, pred_scores)
    AUC = roc_auc_score(y_test, pred_scores)
    return curve, AUC

def get_preds_df(model, X_test, y_test, device='cuda'):
    if model.__class__.__name__.startswith('MLP'):
        model.eval()
        model.to(device)
        X_test = X_test.to(device)
        with torch.no_grad():
            scores = (F.sigmoid(model(X_test).detach().cpu()).view(-1,1))
            preds = quantize(scores, threshold=0.5).numpy()
            scores = scores.numpy()

    else: #SKLearn wrapper
        scores = model.predict_proba(X_test)[:,1]
        preds = model.predict(X_test)
    #print(y_test.shape, scores.shape, preds.shape)
    tmp_data = np.concatenate([y_test.reshape(-1,1), preds.reshape(-1,1), scores.reshape(-1,1)], axis=1)
    
    df = pd.DataFrame(data=tmp_data, columns =['y_true','predicted','positive_score'])

    df['tp'] = df.apply(lambda x: 1 if (x['y_true']==x['predicted'] and x['predicted']==1) else 0, axis=1)
    df['fp'] = df.apply(lambda x: 1 if (x['y_true']!=x['predicted'] and x['predicted']==1) else 0, axis=1)
    df['tn'] = df.apply(lambda x: 1 if (x['y_true']==x['predicted'] and x['predicted']==0) else 0, axis=1)
    df['fn'] = df.apply(lambda x: 1 if (x['y_true']!=x['predicted'] and x['predicted']==0) else 0, axis=1)

    df=df.astype({'y_true':'int64','predicted':'int64', 'positive_score':'float64',
                 'tp':'int64','fp':'int64','tn':'int64','fn':'int64'}, copy=True)
    return df

def get_PPV_curves(df):
    xs=np.cumsum(df.sort_values('positive_score',ascending=False)['tp'].values)/np.cumsum(np.ones(len(df)))

    perfect = np.cumsum(df.sort_values('y_true',ascending=False)['y_true'].values/np.cumsum(np.ones(len(df))))

    return xs, perfect 

def kfold_cv(clf, X, y, k=5, random_state = 20, **kwargs):
    kf = KFold(k, shuffle = True, random_state= random_state)
    roc_dict = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f'Fold number {fold+1}:')
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        #if clf == NN
        if clf.__class__.__name__.startswith('MLP_'):
            clf.reset_params()
            clf.zero_grad()
            train_clf(model = clf, data = (X_train, X_test, X_test, y_train, y_test, y_test),
                      **kwargs)
            (curve, AUC) = reload_predict(model = clf, data = (X_train, X_test, X_test, y_train, y_test, y_test),
                                          **kwargs)
                
            preds_df = get_preds_df(clf, X_test,y_test)
            xs, perfect = get_PPV_curves(preds_df)
            roc_dict[fold] = (curve, AUC, xs, perfect)
            
        if clf.__class__.__name__ == 'XGBClassifier':
            #Blanking params by cloning from base class 
            clf_clone = sklearn.base.clone(clf)
            clf_clone.fit(X_train, y_train, verbose = False,
                     eval_metric = 'aucpr',
                     eval_set = [(X_test, y_test)], **kwargs)
            
            scores = clf_clone.predict_proba(X_test)[:,1]
            curve = roc_curve(y_test, scores)
            AUC = roc_auc_score(y_test, scores)
            print(f'At fold {fold+1}, Test AUC: {AUC}')
            
            preds_df = get_preds_df(clf_clone, X_test,y_test)
            xs, perfect = get_PPV_curves(preds_df)
            
            roc_dict[fold] = (curve, AUC, xs, perfect)
            
        if not clf.__class__.__name__.startswith('MLP') and clf.__class__.__name__ != 'XGBClassifier': 
            #case where clf is from sklearn with its API
            #Blanking params by cloning from base class 
            clf_clone = sklearn.base.clone(clf)
            clf_clone.fit(X_train, y_train)
            scores = clf_clone.predict_proba(X_test)[:,1]
            curve = roc_curve(y_test, scores)
            AUC = roc_auc_score(y_test, scores)
            print(f'Length of test set:\t\t{len(y_test)}\nlen of prediction scores:\t{len(scores)}\nlength of roc_curve array:\t{len(curve[0])}\n')
            print(f'At fold {fold+1}, Test AUC: {AUC}')
            preds_df = get_preds_df(clf_clone, X_test,y_test)
            xs, perfect = get_PPV_curves(preds_df)
            roc_dict[fold] = (curve, AUC, xs, perfect)
            
    return roc_dict

