import pandas as pd
import json
from tqdm.auto import tqdm
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from torch import nn
from torch.nn import SELU, LeakyReLU, ELU, ReLU
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_vae_loss_accs, \
    get_dict_of_lists, get_class_initcode_keys
from src.torch_utils import load_checkpoint, save_model_full, save_json, load_json, load_model_full
from src.models import FullTCRVAE
import glob
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def fix_json(directory):
    files = glob.glob(f'{directory}*')
    pt_file = next(filter(lambda x: 'checkpoint' in x and x.endswith('.pt'), files))

    try:
        js_file = next(filter(lambda x: 'checkpoint' in x and x.endswith('.json'), files))
        js_dict = load_json(js_file)
    except:
        args = next(filter(lambda x: 'args' in x and x.endswith('.txt'), files))
        with open(args, 'r') as f:
            args = f.readlines()
        args = [x.replace('\n','').replace('\t',' ').split(': ') for x in args]
        args_dict = {x[0]:x[1] for x in args if len(x) == 2}
        model_keys = get_class_initcode_keys(FullTCRVAE, args_dict)
        kwargs = {k:args_dict[k] for k in model_keys}
        for k in kwargs:
            if k =='activation':
                kwargs[k] = eval(kwargs[k]).__class__.__name__
            elif k=='pad_scale':
                kwargs[k] = float(kwargs[k])
            elif k=='encoding':
                kwargs[k] = str(kwargs[k])
            else:
                kwargs[k] = int(kwargs[k])
        kwargs['constructor']='FullTCRVAE'
        js_file = pt_file.replace('.pt', '.json').replace('checkpoint','checkpoint_FIXED_KWARGS')
        try:
            save_json(kwargs, js_file)
            model=load_model_full(pt_file, js_file)
        except:
            os.remove(js_file)
            raise ValueError('Couldn\'t load model with the "fixed" kwargs!')

    return 0


# Assuming all files are moved to a main directory (like TripletTest)
maindir=sys.argv[1]
subdirs=glob.glob(f'{maindir}/*/')
alldirs=[glob.glob(f'{sub}/*/') for sub in subdirs]
alldirs=[x for y in alldirs for x in y]
for f in alldirs:
    fix_json(f)
# Parallel(n_jobs=6)(delayed(fix_json)(directory=x) for x in tqdm(alldirs))

