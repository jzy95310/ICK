# Reference: Hatt, Tobias, and Stefan Feuerriegel. "Estimating average treatment effects via orthogonal 
# regularization." Proceedings of the 30th ACM International Conference on Information & Knowledge 
# Management. 2021.

import torch
from torch import nn
from .cfrnet import DenseCFRNet, Conv2DCFRNet
from kernels.nn import ImplicitConvNet2DKernel

class DenseDONUT(DenseCFRNet):
    """
    PyTorch implementation of Deep Orthogonal Networks for Unconfounded Treatments (DONUT) from 
    Hatt and Stefan (2021) with dense layers in representation network. The network use the TARNet/CFRNet
    architecture for the outcome model. Only binary treatment is supported.

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
    skip_connection: bool, whether to use skip connection in the representation and hypothesis 
        networks, default to False
    """
    def __init__(self, input_dim: int, phi_depth: int, phi_width: int, h_depth: int, h_width: int, 
                 activation: str = 'elu', dropout_ratio: float = 0.0, skip_connection: bool = False):
        super(DenseDONUT, self).__init__(input_dim, phi_depth, phi_width, h_depth, h_width,
                                         activation, dropout_ratio, skip_connection)
        self.logreg = nn.Sequential(
            nn.Linear(input_dim, 1), 
            nn.Sigmoid()
        )
        self.epsilon = nn.Parameter(torch.zeros(1,1))
        nn.init.normal_(self.epsilon, mean=0.0, std=0.05)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        --------------
        X: torch.Tensor, the input data

        Returns
        --------------
        concat_pred: torch.Tensor, a concatenated tensor which consists of y0_pred, y1_pred,
            t_pred, and the model parameter epsilon
        """
        phi_X = self.phi(X)
        y0_pred = self.hs[0](phi_X)
        y1_pred = self.hs[1](phi_X)
        t_pred = self.logreg(X)
        epsilon = self.epsilon * torch.ones_like(t_pred)
        concat_pred = torch.cat((y0_pred, y1_pred, t_pred, epsilon), dim=1)
        return concat_pred

class Conv2DDONUT(Conv2DCFRNet):
    """
    PyTorch implementation of Deep Orthogonal Networks for Unconfounded Treatments (DONUT) from 
    Hatt and Stefan (2021) with convolutional layers in representation network. The network use the 
    TARNet/CFRNet architecture for the outcome model. Only binary treatment is supported.

    Arguments
    --------------
    Arguments
    --------------
    input_width: int, the width of the input data
    input_height: int, the height of the input data
    in_channels: int, the number of channels of the input data
    batch_norm: bool, whether to use batch normalization in the representation network, 
        default to False
    """
    def __init__(self, input_width: int, input_height: int, in_channels: int, phi_depth: int, phi_width: int, 
                 h_depth: int, h_width: int, activation: str = 'relu', dropout_ratio: float = 0.0, 
                 batch_norm: bool = False, skip_connection: bool = False, propensity_net_width: int = 128) -> None:
        super(Conv2DDONUT, self).__init__(input_width, input_height, in_channels, phi_depth, phi_width, 
                                          h_depth, h_width, activation, dropout_ratio, batch_norm, 
                                          skip_connection)
        self.propensity_model = nn.Sequential(
            ImplicitConvNet2DKernel(
                input_width=input_width,
                input_height=input_height,
                in_channels=in_channels,
                latent_feature_dim=1,
                num_blocks=1,
                activation=activation,
                num_hidden_dense_layers=1,
                num_dense_units=propensity_net_width, 
                skip_connection=skip_connection
            ), 
            nn.Sigmoid()
        )
        self.epsilon = nn.Parameter(torch.zeros(1,1))
        nn.init.normal_(self.epsilon, mean=0.0, std=0.05)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        --------------
        X: torch.Tensor, the input data

        Returns
        --------------
        concat_pred: torch.Tensor, a concatenated tensor which consists of y0_pred, y1_pred,
            t_pred, and the model parameter epsilon
        """
        phi_X = self.phi(X)
        y0_pred = self.hs[0](phi_X)
        y1_pred = self.hs[1](phi_X)
        t_pred = self.propensity_model(X)
        epsilon = self.epsilon * torch.ones_like(t_pred)
        concat_pred = torch.cat((y0_pred, y1_pred, t_pred, epsilon), dim=1)
        return concat_pred