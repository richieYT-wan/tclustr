import os
from .utils import mkdirs
from torch.nn import LeakyReLU, ELU, SELU, ReLU
import json
from src.models import *

ACT_DICT = {'SELU': nn.SELU(), 'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(), 'ELU': nn.ELU()}
# TODO HERE ADD ALSO FOR DATASET THING (be smarter about this)

def load_model_full(checkpoint_filename, json_filename, dir_path=None):
    """
    Instantiate and loads a model directly from a checkpoint and json filename
    Args:
        checkpoint_filename:
        json_filename:
        dir_path:
    Returns:

    """
    dict_kwargs = load_json(json_filename, dir_path)
    assert 'constructor' in dict_kwargs.keys(), f'No constructor class name provided in the dict_kwargs keys! {dict_kwargs.keys()}'
    constructor = dict_kwargs.pop('constructor')
    dict_kwargs['activation'] = eval(dict_kwargs['activation'])()
    model = eval(constructor)(**dict_kwargs)
    model = load_checkpoint(model, checkpoint_filename, dir_path)
    return model


def save_model_full(model, checkpoint_filename='checkpoint.pt', dir_path='./', verbose=False, best_dict=None,
                    dict_kwargs=None, json_filename=None):
    """
    Saves a torch model (.pt) along with its init parameters in a JSON file
    Args:
        model: Model object
        checkpoint_filename:
        dir_path:
        verbose:
        best_dict:
        dict_kwargs:
        json_filename:

    Returns:

    """
    if json_filename is None:
        json_filename = f'{checkpoint_filename.split(".pt")[-2]}_JSON_kwargs.json' if checkpoint_filename.endswith(
            '.pt') \
            else f'{checkpoint_filename}_JSON_kwargs.json'
    save_checkpoint(model, checkpoint_filename, dir_path, verbose, best_dict)
    if 'constructor' not in dict_kwargs.keys():
        dict_kwargs['constructor'] = model.__class__.__name__
    save_json(dict_kwargs, json_filename, dir_path)
    print(
        f'Model weights saved at {os.path.abspath(os.path.join(dir_path, checkpoint_filename))} ' \
        f'and JSON at {os.path.abspath(os.path.join(dir_path, json_filename))}')


def load_json(filename, dir_path=None):
    """
    Loads a dictionary from a .json file and returns it
    Args:
        filename:

    Returns:
        dict_kwargs: A dictionary containing the kwargs necessary to instantiate a given model
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    with open(filename, 'r') as json_file:
        dict_kwargs = json.load(json_file)
    return dict_kwargs


def save_json(dict_kwargs, filename, dir_path='./'):
    """
    Saves a dictionary to a .json file
    When saving a model, should try to ensure that the model's constructor // class name exists
    Args:
        dict_kwargs:
        filename:
        dir_path:

    Returns:

    """
    savepath = os.path.join(dir_path, filename)
    dict_kwargs['activation'] = dict_kwargs['activation'].__class__.__name__
    # Write the dictionary to a JSON file
    with open(savepath, 'w') as json_file:
        json.dump(dict_kwargs, json_file)
    print(f"JSON data has been written to {savepath}")


def save_checkpoint(model, filename: str = 'checkpoint.pt', dir_path: str = './',
                    verbose=False,
                    best_dict: dict = None):
    """
    Saves a single torch model, with some sanity checks
    Args:
        model: torch model (i.e. anything that inherits from nn.Module and has a state_dict())
        name: the name itself (ex: CNN_model_t0_v1 for a CNN model from the test fold 0, validation fold 1)
        dir_path: path to the directory where the models should be saved

    Returns:
        nothing.
    """
    # Small sanity checks
    assert hasattr(model, 'state_dict'), f'Object of type {type(model)} has no state_dict and can\'t be saved!'
    # Bad practice but fuck it
    if not os.path.exists(dir_path):
        mkdirs(dir_path)
        if verbose:
            print(f'Creating {dir_path}; The provided dir path {dir_path} did not exist!')

    savepath = os.path.join(dir_path, filename)
    checkpoint = model.state_dict()
    if best_dict is not None and type(best_dict) == dict:
        checkpoint['best'] = best_dict
    torch.save(checkpoint, savepath)
    if verbose:
        print(f'Model saved at {os.path.abspath(savepath)}')


def load_checkpoint(model, filename: str, dir_path: str = None, verbose=True):
    """
    Loads a model
    Args:
        model:
        name:
        dir_path:

    Returns:
        model: Loaded model, in eval mode.
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    try:
        checkpoint = torch.load(filename)
        if 'best' in checkpoint.keys():
            best = checkpoint.pop('best')
            if verbose:
                print('Reloading best model:')
                for k, v in best.items():
                    print(f'{k}: {v}')
        model.load_state_dict(checkpoint)
    except:
        st = torch.load(filename)
        print(st.keys())
        raise ValueError()
    model.eval()
    return model


def create_load_model(constructor, filename: str, dir_path: str = None):
    """If provided with a constructor, loads the state_dict and creates the model object from the state_dict

    Assumes that state_dict has a key "init_params" that is used to initialize the model with the constructor.
    Args:
        constructor:
        filename:
        dir_path:

    Returns:
        model: Loaded model, in eval mode, created from the constructor + the state_dict['init_params']
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    state_dict = torch.load(filename)
    assert 'init_params' in state_dict.keys(), f'state_dict does not contain init_params key! It has {state_dict.keys()} instead.'
    model = constructor(**state_dict['init_params'])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def set_mode(models_dict, mode='eval'):
    """
    QOL function to set all models to train or eval
    Args:
        models_dict (dict): dictionary of list of models
        mode (str): mode.lower() should be either 'eval' or 'train'
    Returns:
        the same models_dict
    """
    assert mode.lower() in ['eval', 'train'], 'Please provide a proper mode' \
                                              f'(either "train" or "eval"). You provided "{mode}"'
    mode_bool = mode.lower() == 'train'
    for key, model_list in models_dict.items():
        for model in model_list:
            model.train(mode_bool)


def set_device(models_list, device):
    """
    QOL fct to set all models in a LIST to the same device.
    Only for list and not Dict because copying takes time and space
    Args:
        models_list:
        device:

    Returns:

    """
    for model in models_list:
        model.to(device)
