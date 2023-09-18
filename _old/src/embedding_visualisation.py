import pandas as pd
from vae_cel.vae_cel import *
from vae_cel.vae_cel_loss import *
from vae_cel.vae_cel_train import *
from src.torch_util import load_model
from src.preprocessing import * 

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 300
sns.set_style('darkgrid')

torch.manual_seed(20)

def plot_latent(df, cols, hue):
    #vars_= [z for z in df.columns if z.startswith('PCA')]
    vars_ = cols
    print(f'VAE latent space representation for {cols[0]}-{cols[-1]}')
    g = sns.PairGrid(df, x_vars = vars_,
                     y_vars = vars_,
                     diag_sharey=False, hue=hue)
    g.fig.suptitle(f'VAE latent space representation for {cols[0]}-{cols[-1]}', fontweight='bold')
    g.fig.subplots_adjust(top=0.92)
    g.map_lower(sns.scatterplot,  alpha=0.75)
    g.map_upper(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.add_legend()
    plt.show()
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
           
def get_embedding_df(model, dataset, device):
    """takes a model and dataset, embed the data in latent dim and returns the df"""
    model.to(device)
    chks = chunks(range(len(dataset)), 2**16)
    df = pd.DataFrame(columns = (['sequence'] + ['z_'+str(x) for x in list(range(model.lat_dim))]))
    model.eval()
    print('embedding in latent space')
    for b in tqdm(list(chks), desc = 'batch'): 
        with torch.no_grad():
            x_tuple = batch_aa_vj(dataset[b], 23, 0.5, 'before', True, False,device)
            z = model.embed(x_tuple).detach().cpu().numpy()
        df_s = pd.DataFrame(dataset[b], columns = ['sequence','v','j'])
        df_z = pd.DataFrame(z, columns = ['z_'+str(x) for x in list(range(model.lat_dim))])
        df = df.append(df_s.join(df_z), ignore_index=True)
    return df

           
def get_AE_embedding_df(model, dataset, device):
    """takes a model and dataset, embed the data in latent dim and returns the df"""
    model.to(device)
    chks = chunks(range(len(dataset)), 2**16)
    df = pd.DataFrame(columns = (['sequence'] + ['z_'+str(x) for x in list(range(model.lat_dim))]))
    model.eval()
    print('embedding in latent space')
    for b in tqdm(list(chks), desc = 'batch'): 
        with torch.no_grad():
            x = onehot_batch(dataset[b], 23, 1.6, 'after', False, False).to(device)
            z = model.embed(x).detach().cpu().numpy()
        df_s = pd.DataFrame(dataset[b], columns = ['sequence'])
        df_z = pd.DataFrame(z, columns = ['z_'+str(x) for x in list(range(model.lat_dim))])
        df = df.append(df_s.join(df_z), ignore_index=True)
    return df

def get_patients(n=2, show =True):
    print(f'\t Getting {n} patients')
    fn = np.random.choice(sample_tags.filename.unique(), n)
    patients_queried = emerson.query('filename in @fn')
    print(f'Length of df {len(patients_queried)}')
    if show == True:
        display(sample_tags.query('filename in @fn')[['filename','age', 'sex', 'race', 'hla_a1',
                                                      'hla_a2', 'hla_b1', 'hla_b2', 'class']])
        aas = patients_queried.query('filename==@fn[0]').amino_acid.values
        print("Total shared : ", len(patients_queried.query('filename==@fn[1] and amino_acid in @aas')))
    return patients_queried.query('len>10 and len<=23')

def get_topk(df, topk=2500, verbose=True):
    print(f'\t Getting the top {topk}')
    top = df.sort_values('frequency', ascending=False)\
            .groupby('filename').head(topk)
    fn_ = top.filename.unique()
    if verbose:
        aas = top.query("filename in @fn_[1]").amino_acid.values
        len_shared = len(top.query('filename==@fn_[0] and amino_acid in @aas'))
        print(f"# Shared in top {topk} seqs : {len_shared}")
        return top, len_shared
    else:
        return top
    
def get_reduc(df, model, reduc='TSNE', n_c = 2):
    if reduc == 'TSNE':
        reducer = TSNE(n_components = n_c, random_state=42)
    elif reduc == 'PCA':
        reducer = PCA(n_components = n_c, random_state = 42)
    elif reduc == 'UMAP':
        reducer = UMAP(n_components = n_c, random_state=42)
    
    df['v'] = df['v_family'].apply(lambda x : int(x.split('V')[1]) -1)
    df['j'] = df['j_family'].apply(lambda x : int(x.split('J')[1]) -1)
    
    model_device = next(model.children())[0].weight.device
    print('\t Onehot-encoding.')
    model.eval()
    x_tuple = batch_aa_vj(df[['amino_acid','v','j']].values, 23, 0.5, 'before', True, False, model_device)
    
    print(f'\t Embedding {len(df)} sequences.')
    with torch.no_grad():
        z = model.embed(x_tuple).detach().cpu().numpy()
        
    print(f'\t Fitting {reduc}.')
    results = reducer.fit_transform(z)
    print('\t Waiting on plotting.')
    for i in range(results.shape[1]):
        df[f'{reduc}_'+str(i+1)] = results[:,i]
    return df

def plot_reduc(df, hue, name, reduc = 'TSNE', palette = 'seismic_r', comp = True):
    #if hue is not None:
    unique = len(df[hue].unique())
    #else :
    #    unique = 10
    sns.set_palette(palette, unique)
    MARKERS = ['X','P','D','s','o','*']
    if unique > 6:
        MARKERS = ['X']*unique
    vars_= [z for z in df.columns if z.startswith(reduc)]
    g = sns.PairGrid(df, x_vars = vars_,
                     y_vars = vars_,
                     diag_sharey=False, hue=hue,
                     hue_kws = {'markers':MARKERS[:unique]})
    if comp == True : 
        title = f'{name} TCR, {reduc} with {len(vars_)} components'
        adjust = 0.91
    elif comp == False:
        classes = '//'.join(sample_tags.query('filename in @df.filename.unique()')['class'].values)
        title = f'VAE : {name} TCR, {reduc} with {len(vars_)} components\n'\
                   f'HLA Classes : {classes}'
        adjust = 0.89
        
    g.fig.suptitle(title, fontweight='bold')
    g.fig.subplots_adjust(top=adjust)
    
    g.map_lower(sns.scatterplot,  alpha=0.8, size = df[hue],
                markers= MARKERS[:unique], style = df[hue],
                sizes=[3]*unique)
    g.map_upper(sns.kdeplot, levels = 8, alpha = 0.75)
    g.map_diag(sns.kdeplot)
    g.add_legend()
    
    
def pipeline(df, model, topk, reduc='TSNE', n_c=2, hue = 'filename', name = '',
             palette = 'seismic_r', comp = True):
    """
    from an already queried df (like which patients), 
    embed, reduc, plot etc
    """
    top = get_topk(df, topk, verbose = False)
    df_reduc = get_reduc(top, model, reduc, n_c)
    plot_reduc(df_reduc, hue, name, reduc, palette, comp)
    return df_reduc
    
    