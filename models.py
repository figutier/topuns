import math

import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured, random_unstructured, global_unstructured, L1Unstructured, RandomUnstructured
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn.functional as F



class MLP(nn.Module):
    """Multilayer perceptron.

    The bias is included in all linear layers.

    Parameters
    ----------
    n_features : int
        Number of input features (pixels inside of MNIST images).

    hidden_layer_sizes : tuple
        Tuple of ints representing sizes of the hidden layers.

    n_targets : int
        Number of target classes (10 for MNIST).

    Attributes
    ----------
    module_list : nn.ModuleList
        List holding all the linear layers in the right order.
    """
    
    


    def __init__(self, n_features, hidden_layer_sizes, n_targets, seed = None):
        super().__init__()
        if seed != None:
            torch.manual_seed(seed)
        layer_sizes = (n_features,) + hidden_layer_sizes + (n_targets,)
        layer_list = []

        for i in range(len(layer_sizes) - 1):
            layer_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of features of shape `(batch_size, n_features)`.

        Returns
        -------
        torch.Tensor
            Batch of predictions (logits) of shape `(batch_size, n_targets)`.
        """
        
        n_layers = len(self.module_list)


        for i, layer in enumerate(self.module_list):
            
            x = layer(x)

            if i < n_layers - 1:
                x = nn.functional.relu(x)

        return x