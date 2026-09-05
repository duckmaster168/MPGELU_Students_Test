import torch
import os
from typing import Iterable, Callable
import sys
from pathlib import Path

###Load our model
from MyCNN import MyCNN

from MyCNN import MyCNN
class LoadedModel:
    def __init__(self,
                 activation_dict: dict,
                 model_params: list,
                 device: str,
                 checkpoint_dir = "model_trained_params"):
        self.activation_dict = activation_dict
        self.model_params = model_params
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.load_models = {}
    def load_checkpoints(self):
        for act_name, act_fn in self.activation_dict.items():
            print(f"\n=======================================================")
            print(f"       EVALUATING ACTIVATION: {act_name}")
            print(f"=======================================================")
            model = MyCNN(input_shape=3, 
                        output_shape=10, 
                        activation=act_fn, 
                        params=self.model_params).to(self.device)
            path_to_trained_acn = f"{self.checkpoint_dir}/chkpt_{act_name}.pth"
            if os.path.exists(path_to_trained_acn):
                trained_model = torch.load(path_to_trained_acn, map_location=self.device, weights_only=False)
                print(f"Loaded trained model for {act_name} from {path_to_trained_acn}")
                model.load_state_dict(trained_model['model_state_dict'])
                model.eval()
                self.load_models[act_name] = {
                "model": model,
                "results": trained_model['results'],
                "total_runtime": trained_model['total_runtime'],
                "activation_name": trained_model['activation_name']
            }
            else:
                print(f"Walang nakita")
        return self.load_models