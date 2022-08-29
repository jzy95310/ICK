# bnn.py: a file containing the definition of implicit kernel implied by Bayesian neural networks
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn
import torchbnn as bnn
from .nn import ImplicitDenseNetKernel
from .constants import ACTIVATIONS

class ImplicitDenseBayesNetKernel(ImplicitDenseNetKernel):
    """
    Implicit kernel implied by a dense Bayesian neural network

    Arguments
    --------------
    prior_mean: float, the mean of the Gaussian prior of weights
    prior_std: float, the standard deviation of the Gaussian prior of weights
    """
    def __init__(self, input_dim: int, latent_feature_dim: int, num_blocks: int, num_layers_per_block: int, 
                 num_units: int, activation: str = 'relu', dropout_ratio: float = 0.0, prior_mean: float = 0.0, 
                 prior_std: float = 1.0) -> None:
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        super(ImplicitDenseBayesNetKernel, self).__init__(input_dim, latent_feature_dim, num_blocks, num_layers_per_block, 
                                                          num_units, activation, dropout_ratio)
        self._validate_inputs()
        self.build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.prior_std > 0.0, "The standard deviation of the Gaussian prior of weights should be positive."
        super(ImplicitDenseBayesNetKernel, self)._validate_inputs()
    
    def build_layers(self) -> None:
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        if self.num_blocks == 0:
            self.dense_blocks.append(bnn.BayesLinear(self.prior_mean, self.prior_std, self.input_dim, self.latent_feature_dim))
        else:
            self.dense_blocks.append(
                nn.Sequential(
                    bnn.BayesLinear(self.prior_mean, self.prior_std, self.input_dim, self.num_units),
                    ACTIVATIONS[self.activation],
                    nn.Dropout(self.dropout_ratio)
                )
            )
            for _ in range(self.num_blocks - 1):
                self.dense_blocks.append(
                    nn.Sequential(
                        *[bnn.BayesLinear(self.prior_mean, self.prior_std, self.num_units, self.num_units) \
                            for _ in range(self.num_layers_per_block)],
                        ACTIVATIONS[self.activation],
                        nn.Dropout(self.dropout_ratio)
                    )
                )
            self.dense_blocks.append(bnn.BayesLinear(self.prior_mean, self.prior_std, self.num_units, self.latent_feature_dim))