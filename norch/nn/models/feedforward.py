import torch.nn as nn
import torch
from ..layers import bn_drop_lin

class  FeedforwardNeuralNetModel (nn.Module):
    def __init__(self, input_dim : int , hidden_dims : list, num_classes : int,
                 bn : bool, drop: float):
        """
        Simple Feedforward Net that accpest variable amount of layers
        
        Arguments : 
            input_dim : Size of the input dimension
            hidden_dims : List containing size of all hidden layers
            num_classes : Number of classes
            bn : If there is a batch norm layer
            drop : dropout rate
        """
        super(FeedforwardNeuralNetModel, self).__init__()
        
        layer_size = [input_dim] +hidden_dims +[num_classes]
        
        layers = []
        for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
            if n_out != hidden_dims :
                #add ReLU for every layer except last one
                layers.append(bn_drop_lin(n_in, n_out, bn, drop, nn.ReLU()))
            else : 
                #don't add ReLU to last layer
                layers.append(bn_drop_lin(n_in, n_out, bn, drop, None))
        self.layers = nn.Sequential(layers)
        
        
    
    def forward(self, x):
        # Linear function 1
        out = self.layers(x)
        return out
