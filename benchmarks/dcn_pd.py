# Reference: Alaa, Ahmed M., Michael Weisz, and Mihaela Van Der Schaar. "Deep counterfactual 
# networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017).

import torch
from torch import nn
from kernels.nn import ImplicitDenseNetKernel

class DenseDCNPD(nn.Module):
    """
    PyTorch implementation of the Deep Counterfactual Network with Propensity Dropout (DCN-PD) from
    Alaa et al. (2017). Only binary treatment is supported.

    Arguments
    --------------
    shared_depth: int, the number of shared layers
    shared_width: int, the number of units in each shared layer
    idiosyncratic_depth: int, the number of idiosyncratic layers for treatment and control
    idiosyncratic_width: int, the number of units in each idiosyncratic layer
    pd_net_depth: int, the number of layers in the propensity dropout network
    pd_net_width: int, the number of units in each layer of the propensity dropout network
    activation: str, the activation function 
    """
    def __init__(self, shared_depth: int, shared_width: int, idiosyncratic_depth: int, idiosyncratic_width: int, 
                 pd_net_depth: int, pd_net_width: int, activation: str = 'relu'):
        super(DenseDCNPD, self).__init__()
        self.shared_layers = ImplicitDenseNetKernel(
            
        )