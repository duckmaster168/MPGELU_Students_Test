import torch as torch
import torch.nn.functional as F
from typing import Iterable, Callable

class MyCNN(torch.nn.Module):
    '''Custom CNN model with flexible architecture'''
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyCNN, self).__init__()
        self.input_shape = input_shape
        self.activation = activation
        self.conv_layers = self.convo_layers(params)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*4*4,
                    out_features=256),
            self.activation(),
            torch.nn.Linear(in_features=256,
                    out_features=256),
            self.activation(),
            torch.nn.Linear(in_features=256,
                    out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def convo_layers(self, arch):
        Mylayers = []
        for k in arch:
            if type(k) == int:
                output_channels = k

                Mylayers += [torch.nn.Conv2d(self.input_shape, output_channels, 
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     self.activation()]
                self.input_shape = k
            elif k == "MaxPool":
                Mylayers += [torch.nn.MaxPool2d(kernel_size=(2,2))]
        return torch.nn.Sequential(*Mylayers)
    