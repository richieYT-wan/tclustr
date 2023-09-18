import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from os import makedirs, remove
from os.path import exists, join, basename, dirname

class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        
    def __call__(self, batch):
        batch_var = {}
        for key,value in batch.items():
            if isinstance(value,torch.Tensor) and not self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False)
            elif isinstance(value,torch.Tensor) and self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False).cuda()
            else:
                batch_var[key] = value            
        return batch_var

def load_model(model, path_to_weight):
    """wrapper to do the load_state_dict thing at once"""
    tmp = torch.load(path_to_weight)
    model.load_state_dict(tmp['state_dict'])
    print(f'Model succesfully loaded from {path_to_weight}:')
    for k in [k for k in tmp.keys() if k != 'state_dict']:
        print(f'\t{k}: {tmp[k]}')
    return model

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))
        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))
        