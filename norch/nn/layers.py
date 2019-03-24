"""
Custom layers or sequence of layers
"""
import torch


def bn_drop_lin(
    n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: torch.nn.Module = None,
    sequential : bool = False,
):
    """
    Utility function that adds batch norm, dropout and linear layer 
    
    Arguments : 
        n_in : Number of input neurons
        n_out : Number of output neurons
        bn : If there is a batch norm layer
        p : Bathc norm dropout rate
        act : Activation for the linear layer
    
    Returns : 
        List of batch norm, dropout and linear layer
    
    """
    layers = [torch.nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(torch.nn.Dropout(p))
    layers.append(torch.nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    if sequential :
        return torch.nn.Sequential(layers)
    else :
        return layers