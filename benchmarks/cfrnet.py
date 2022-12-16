# Reference: Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment 
# effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017.
# Link: https://proceedings.mlr.press/v70/shalit17a.html

import torch
from torch import nn
from kernels.nn import ImplicitDenseNetKernel, ImplicitConvNet2DKernel

class DenseCFRNet(nn.Module):
    """
    PyTorch implementation of the Counterfactual Regression Network (CFRNet) from Shalit et al. (2017)
    with dense layers in representation network. Only binary treatment is supported.

    Arguments
    --------------
    input_dim: int, the dimension of the input data
    phi_depth: int, the number of layers in the representation network Phi
    phi_width: int, the number of units in each layer of the representation network Phi
    h_depth: int, the number of layers in the hypothesis network h
    h_width: int, the number of units in each layer of the hypothesis network h
    activation: str, the activation function to use in the representation and hypothesis networks, 
        default to 'relu'
    dropout_ratio: float, the dropout ratio to use in the representation and hypothesis networks,
        default to 0.0
    """
    def __init__(self, input_dim: int, phi_depth: int, phi_width: int, h_depth: int, h_width: int, 
                 activation: str = 'relu', dropout_ratio: float = 0.0):
        super(DenseCFRNet, self).__init__()
        self.phi = ImplicitDenseNetKernel(
            input_dim=input_dim, 
            latent_feature_dim=phi_width, 
            num_blocks=phi_depth, 
            num_layers_per_block=1,
            num_units=phi_width,
            activation=activation, 
            dropout_ratio=dropout_ratio
        )
        self.hs = nn.ModuleList()
        for _ in range(2):
            self.hs.append(
                ImplicitDenseNetKernel(
                    input_dim=phi_width, 
                    latent_feature_dim=1, 
                    num_blocks=h_depth, 
                    num_layers_per_block=1,
                    num_units=h_width,
                    activation=activation, 
                    dropout_ratio=dropout_ratio
                )
            )

    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        --------------
        X: torch.Tensor, the input data
        T: torch.Tensor, the treatment assignment

        Returns
        --------------
        y: torch.Tensor, the predicted outcome
        phi_X: torch.Tensor, the output of the representation network Phi(x)
        """
        phi_X = self.phi(X)
        y = (1 - T) * self.hs[0](phi_X) + T * self.hs[1](phi_X)
        return y, phi_X
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        --------------
        X: torch.Tensor, the input data

        Returns
        --------------
        y: torch.Tensor, the predicted outcome for both control and treatment groups with shape
            (batch_size, 2)
        """
        phi_X = self.phi(X)
        y = torch.cat([self.hs[0](phi_X), self.hs[1](phi_X)], dim=1)
        return y

class Conv2DCFRNet(nn.Module):
    """
    Counterfactual Regression Network (CFRNet) from Shalit et al. (2017) with 2D convolutional layers 
    in representation network. Only binary treatment is supported.

    Arguments
    --------------
    input_width: int, the width of the input data
    input_height: int, the height of the input data
    in_channels: int, the number of channels of the input data
    batch_norm: bool, whether to use batch normalization in the representation network, default to False
    """
    def __init__(self, input_width: int, input_height: int, in_channels: int, phi_depth: int, phi_width: int, 
                 h_depth: int, h_width: int, activation: str = 'relu', dropout_ratio: float = 0.0, 
                 batch_norm: bool = False):
        super(Conv2DCFRNet, self).__init__()
        self.phi = ImplicitConvNet2DKernel(
            input_width=input_width,
            input_height=input_height,
            in_channels=in_channels,
            latent_feature_dim=phi_width,
            num_blocks=phi_depth,
            use_batch_norm=batch_norm,
            activation=activation,
            num_dense_units=phi_width,
            dropout_ratio=dropout_ratio
        )
        self.hs = nn.ModuleList()
        for _ in range(2):
            self.hs.append(
                ImplicitDenseNetKernel(
                    input_dim=phi_width, 
                    latent_feature_dim=1, 
                    num_blocks=h_depth, 
                    num_layers_per_block=1,
                    num_units=h_width,
                    activation=activation, 
                    dropout_ratio=dropout_ratio
                )
            )
    
    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        phi_X = self.phi(X)
        y = (1 - T) * self.hs[0](phi_X) + T * self.hs[1](phi_X)
        return y, phi_X
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        phi_X = self.phi(X)
        y = torch.cat([self.hs[0](phi_X), self.hs[1](phi_X)], dim=1)
        return y