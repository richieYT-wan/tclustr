from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import math

from src.preprocessing import *
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import BatchSampler, RandomSampler
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

import os
import torch
import pandas as pd
import numpy as np
import math
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EmersonRepertoire_Dataset(Dataset):
    """
    Dataset class from the Emerson study (batch1), should facilitate how I handle classes etc.
    path should contain all the necessary data (top10k_train.csv, top10k_test.csv, sample_tags)
    """
    
    def __init__(self, path, max_len, istrain=True, top_k = 10000,
                 allele = 'A', pos_class = 'A01'):
        
        if allele != 'A' and allele != 'B':
            raise Exception('Allele must be "A" or "B"!')
            
        if istrain == True : 
            fn = 'emerson_batch1_train_top_20k.csv'
            which = 'train'
        elif istrain == False : 
            fn = 'emerson_batch1_test_top_20k.csv'
            which = 'test'
        if top_k > 20000: top_k = 20000
        #Reading the sample tags with the associated split-df
        tags = pd.read_csv('../training_data_new/emerson_raw/batch1/emerson_batch1_sampletags.tsv', 
                  sep = '\t')\
                 .query('dataset == @which')\
                 .reset_index()

        #Getting the one vs rest label (must be defined at initialization)
        if allele == 'A' : columns = ['hla_a1', 'hla_a2']
        elif allele == 'B' : columns = ['hla_b1', 'hla_b2'] 
        # 1 if either of hla_x1 or hla_x2 is of that label 
        tags['class_label'] = tags.apply(lambda x: 1 if (x[columns[0]] == pos_class or x[columns[1]] == pos_class) else 0, axis = 1)
        
        #Also this is the actual iterable (i.e. one sample = one patient)
        #saving values
        self.patients = tags.filename.values
        self.labels = tags.class_label.values 
        self.len = len(tags)        
        
        #Reading the top_K most frequent (grouped by patient) values from DF 
        #print("here",os.path.join(path, fn))
        #print("there",max_len, top_k)
        df = get_patient_topk_sequences(pd.read_csv(os.path.join(path, fn))\
                                          .query('amino_acid.str.len() <= @max_len'), 
                                        top_k)
        #print('Here??')
        #Loading the values from DF and saving to attribute
        self.seq_filename = df.filename.values 
        #print('there???')
        self.values = df[['amino_acid', 'v_family','j_family']].values
        #print('how about this')
        self.frequency = df.frequency.values
        self.n_per_bag = np.array([len(df.query('filename == @x')) for x in tags.filename])

    def __getitem__(self, index):
        # 1 index = 1 patient
        patient = self.patients[index]
        target = self.labels[index]
        n_per_bag = self.n_per_bag[index]
        #print(patient, type(patient))
        #Getting the sequences associated with the patients
        if type(patient) == str:
            indices = np.where(self.seq_filename == patient)
        else:
            indices = np.empty(0,dtype=np.int64)
            for p in patient:
                indices = np.append(indices, np.where(self.seq_filename == p))
        
        values = self.values[indices] 
        #An input to DeepRC should be ((x_tuple), n_per_bag) , where x_tuple is batch_aa_vj(values)
        #But should probly do the encoding in DeepRC or in the network
        return values, n_per_bag, target
    
    def __len__(self):
        return self.len
    
def load_train_test_repertoire(path, max_len, top_k, allele, pos_class, split_ratio = .7):
    """
    Loads the train and test dataset for Emerson Repertoire (pre-split),
    then splits the train dataset into train and valid dataset w.r.t. split_ratio = the % of data in train set
    """
    train_dataset = EmersonRepertoire_Dataset(path, max_len, True, top_k, allele, pos_class)
    test_dataset = EmersonRepertoire_Dataset(path, max_len, False, top_k, allele, pos_class)
    #Splitting
    print('Splitting into train/val')
    train_len = int(split_ratio*len(train_dataset))
    val_len = len(train_dataset)-train_len
    train, val = random_split(train_dataset, lengths = [train_len, val_len])
    
    return train_dataset, test_dataset, train, val

    