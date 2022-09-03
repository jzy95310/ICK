# bnn.py: a file containing the definition of implicit kernel implied by Bayesian neural networks
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from torch import nn
import torchbnn as bnn
from .nn import ImplicitDenseNetKernel
from .constants import ACTIVATIONS

class ImplicitDenseBayesNetKernel(ImplicitDenseNetKernel):
    """
    Implicit kernel implied by a dense Bayesian neural network (BNN)
    The BNN is learned by directly minimizing the KL divergence between the weight posterior and prior through
    Variational Bayes. In other words, the BNN is learned by directly estimating the posterior distribution of 
    its parameters.

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
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.prior_std > 0.0, "The standard deviation of the Gaussian prior of weights should be positive."
        super(ImplicitDenseBayesNetKernel, self)._validate_inputs()
    
    def _build_layers(self) -> None:
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

# ########################################################################################
# MIT License

# Copyright (c) 2022 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################################