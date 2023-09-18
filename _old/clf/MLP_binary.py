import torch
import torch.nn as nn
import torch.nn.functional as F 

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

    def reset_params(self):
        i = 0
        for x in self.children():
          if x.__class__.__name__ == 'Sequential':
            for layer in x:
              if hasattr(layer, 'reset_parameters'):
                  layer.reset_parameters()
                  layer.zero_grad()
                  i = 1
        if i == 1: print('Parameters reset.')


class MLP_AnotherBinary(nn.Module):
    #Reusing an architecture from Deepcat 
    def __init__(self, in_dim):
        super(MLP_AnotherBinary, self).__init__()
            
        self.layers = nn.Sequential(nn.Linear(in_dim, 256),
                                    nn.BatchNorm1d(256),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(256, 512),
                                    nn.BatchNorm1d(512),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(512, 256),
                                    nn.BatchNorm1d(256),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                   )
        self.output = nn.Sequential(nn.Linear(256, 64),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(64, 10),
                                    nn.SELU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(10,1))

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x.view(-1,)
    
    def reset_params(self):
        i = 0
        for x in self.children():
            if x.__class__.__name__ == 'Sequential':
                for layer in x:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                        layer.zero_grad()
                        i += 1
            if hasattr(x, 'reset_parameters'):
                x.reset_parameters()
                x.zero_grad()
                i += 1
        if i >= 1: print('Parameters reset.')
            