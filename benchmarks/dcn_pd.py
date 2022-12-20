# Reference: Alaa, Ahmed M., Michael Weisz, and Mihaela Van Der Schaar. "Deep counterfactual 
# networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017).

import torch
from torch import nn
from copy import deepcopy
from kernels.nn import ImplicitDenseNetKernel, ImplicitConvNet2DKernel

class DenseDCNPD(nn.Module):
    """
    PyTorch implementation of the Deep Counterfactual Network with Propensity Dropout (DCN-PD) from
    Alaa et al. (2017). Only binary treatment is supported.

    Arguments
    --------------
    input_dim: int, the dimension of the input
    shared_depth: int, the number of shared layers
    shared_width: int, the number of units in each shared layer
    idiosyncratic_depth: int, the number of idiosyncratic layers for treatment and control
    idiosyncratic_width: int, the number of units in each idiosyncratic layer
    pd_net_depth: int, the number of layers in the propensity dropout network
    pd_net_width: int, the number of units in each layer of the propensity dropout network
    activation: str, the activation function, default to 'relu'
    gamma: float, offset hyper-parameter, default to 1.0
    """
    def __init__(self, input_dim: int, shared_depth: int = 2, shared_width: int = 512, idiosyncratic_depth: int = 2, 
                 idiosyncratic_width: int = 512, pd_net_depth: int = 2, pd_net_width: int = 512, activation: str = 'relu', 
                 gamma: float = 1.0):
        super(DenseDCNPD, self).__init__()
        self.gamma = gamma
        self.shared_layers = ImplicitDenseNetKernel(
            input_dim=input_dim,
            latent_feature_dim=shared_width,
            num_blocks=shared_depth-1,
            num_layers_per_block=1,
            num_units=shared_width,
            activation=activation
        )
        self.idiosyncratic_layers = nn.ModuleList()
        for _ in range(2):
            self.idiosyncratic_layers.append(
                ImplicitDenseNetKernel(
                    input_dim=shared_width,
                    latent_feature_dim=1,
                    num_blocks=idiosyncratic_depth-1,
                    num_layers_per_block=1,
                    num_units=idiosyncratic_width,
                    activation=activation
                )
            )
        self.pd_net = ImplicitDenseNetKernel(
            input_dim=input_dim,
            latent_feature_dim=1,
            num_blocks=pd_net_depth,
            num_layers_per_block=1,
            num_units=pd_net_width,
            activation=activation
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to only both shared and idiosyncratic layers in forward pass, only for training
        and validation but not for testing

        Arguments
        --------------
        X: torch.Tensor, the input data
        """
        y = torch.zeros(X.shape[0], 2)
        p = torch.sigmoid(self.pd_net(X))   # shape: (batch_size, 1)
        p_dropout = 1 - self.gamma/2 - 1/2*(-p*torch.log(p) - (1-p)*torch.log(1-p))

        for block in self.shared_layers.dense_blocks:
            input_size = block[0].in_features if isinstance(block,torch.nn.Sequential) else block.in_features
            mask = torch.bernoulli(p_dropout.repeat(1,input_size))
            X = block(X * mask)

        shared = [X] * 2
        for i in range(2):
            x = shared[i]
            for block in self.idiosyncratic_layers[i].dense_blocks:
                input_size = block[0].in_features if isinstance(block,torch.nn.Sequential) else block.in_features
                mask = torch.bernoulli(p_dropout.repeat(1,input_size))   # shape: (batch_size, input_size)
                x = block(x * mask)
            y[:,i] = x.squeeze()
        return y
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict potential outcomes without dropout, only for testing

        Arguments
        --------------
        X: torch.Tensor, the input data
        """
        shared = self.shared_layers(X)
        y = torch.cat([self.idiosyncratic_layers[0](shared), self.idiosyncratic_layers[1](shared)], dim=1)
        return y

class Conv2DDCNPD(nn.Module):
    """
    DCN-PD with convolutional layers as the shared layers
    """
    def __init__(self, input_width: int, input_height: int, in_channels: int, shared_conv_blocks: int = 2, 
                 shared_channels: int = 64, shared_output_size: int = 512, idiosyncratic_depth: int = 2,
                 idiosyncratic_width: int = 512, pd_net_conv_blocks: int = 2, pd_net_channels: int = 64, 
                 pd_net_dense_units: int = 128, activation: str = 'relu', gamma: float = 1.0):
        super(Conv2DDCNPD, self).__init__()
        self.gamma = gamma
        self.shared_layers = ImplicitConvNet2DKernel(
            input_width=input_width,
            input_height=input_height,
            in_channels=in_channels,
            latent_feature_dim=shared_output_size,
            num_blocks=shared_conv_blocks,
            num_intermediate_channels=shared_channels,
            activation=activation,
            num_hidden_dense_layers=0
        )
        self.idiosyncratic_layers = nn.ModuleList()
        for _ in range(2):
            self.idiosyncratic_layers.append(
                ImplicitDenseNetKernel(
                    input_dim=shared_output_size,
                    latent_feature_dim=1,
                    num_blocks=idiosyncratic_depth-1,
                    num_layers_per_block=1,
                    num_units=idiosyncratic_width,
                    activation=activation
                )
            )
        self.pd_net = ImplicitConvNet2DKernel(
            input_width=input_width,
            input_height=input_height,
            in_channels=in_channels,
            latent_feature_dim=1,
            num_blocks=pd_net_conv_blocks,
            num_intermediate_channels=pd_net_channels,
            activation=activation,
            num_hidden_dense_layers=1,
            num_dense_units=pd_net_dense_units
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to only the idiosyncratic layers in forward pass, only for training
        and validation but not for testing

        Arguments
        --------------
        X: torch.Tensor, the input data
        """
        y = torch.zeros(X.shape[0], 2)
        p = torch.sigmoid(self.pd_net(X))   # shape: (batch_size, 1)
        p_dropout = 1 - self.gamma/2 - 1/2*(-p*torch.log(p) - (1-p)*torch.log(1-p))
        shared = [self.shared_layers(X)] * 2
        for i in range(2):
            x = shared[i]
            for block in self.idiosyncratic_layers[i].dense_blocks:
                input_size = block[0].in_features if isinstance(block,torch.nn.Sequential) else block.in_features
                mask = torch.bernoulli(p_dropout.repeat(1,input_size))   # shape: (batch_size, input_size)
                x = block(x * mask)
            y[:,i] = x.squeeze()
        return y
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict potential outcomes without dropout, only for testing

        Arguments
        --------------
        X: torch.Tensor, the input data
        """
        shared = self.shared_layers(X)
        y = torch.cat([self.idiosyncratic_layers[0](shared), self.idiosyncratic_layers[1](shared)], dim=1)
        return y