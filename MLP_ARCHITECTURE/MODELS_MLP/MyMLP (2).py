import torch
import torch.nn as nn
from typing import Iterable, Callable

class MyMLP(nn.Module):
    '''
    Custom MLP model with flexible architecture.
    Mirrors the "params list" pattern used in MyCNN.
    '''
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyMLP, self).__init__()
        self.input_shape = input_shape
        self.activation = activation
        
        # Build hidden layers using a local variable so input_shape stays intact
        self.fc_layers, last_dim = self.build_layers(params)

        # Final classification head: maps 512 hidden nodes to 10 class outputs
        self.output_layer = nn.Linear(in_features=last_dim,
                                      out_features=output_shape)

    def forward(self, x):
        # Flatten (N, C, H, W) -> (N, C*H*W), e.g. (N, 1, 28, 28) -> (N, 784)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

    def build_layers(self, arch: Iterable):
        '''
        arch: iterable of ints, e.g. [512, 512, 512, 512]
        '''
        MyLayers = []
        current_dim = self.input_shape
        for num_nodes in arch:
            MyLayers += [nn.Linear(current_dim, num_nodes),
                         self.activation()]
            current_dim = num_nodes
        return nn.Sequential(*MyLayers), current_dim