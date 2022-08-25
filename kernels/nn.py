# nn.py: a file containing the definition of implicit kernel implied by neural networks
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "softmax": nn.Softmax(),
}

class ImplicitNNKernel(nn.Module):
    """
    Parent class of the implicit kernel implied by neural networks

    Arguments
    --------------
    latent_feature_dim: int, the dimension of feature in the latent space
    num_blocks: int, the number of blocks in the neural network, where depth = num_blocks * num_layers
    activation: str, the activation function to be used in each layer, default to 'relu'
    dropout_ratio: float, the dropout ratio for dropout layers, default to 0.0
    """
    def __init__(self, latent_feature_dim: int, num_blocks: int, activation: str = 'relu', dropout_ratio: float = 0.0) -> None:
        super(ImplicitNNKernel, self).__init__()
        self.latent_feature_dim: int = latent_feature_dim
        self.num_blocks: int = num_blocks
        self.activation: str = activation
        self.dropout_ratio: float = dropout_ratio
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the NN-implied implicit kernel class
        """
        assert self.latent_feature_dim > 0, "The number of output features should be positive."
        assert self.num_blocks >= 0, "The number of blocks should be non-negative."
        assert self.activation in ACTIVATIONS, "The activation function should be one of the following: {}".format(ACTIVATIONS.keys())
        assert 0.0 <= self.dropout_ratio <= 1.0, "The dropout ratio should be between 0.0 and 1.0."
    
    def build_layers(self) -> None:
        """
        Build layers of the NN-implied implicit kernel class
        """
        pass
    
    def forward(self) -> None:
        """
        Forward pass of the NN-implied implicit kernel class
        """
        pass

class ImplicitDenseNetKernel(ImplicitNNKernel):
    """
    Implicit kernel implied by a dense neural network

    Arguments
    --------------
    input_dim: int, the dimension of input features
    num_layers: int, the number of dense layers in each HIDDEN block
    num_units: int, the number of units (width) in each HIDDEN dense layer
    """
    def __init__(self, input_dim: int, latent_feature_dim: int, num_blocks: int, num_layers: int, 
                 num_units: int, activation: str = 'relu', dropout_ratio: float = 0.0) -> None:
        self.input_dim = input_dim
        self.num_layers: int = num_layers
        self.num_units: int = num_units
        super(ImplicitDenseNetKernel, self).__init__(latent_feature_dim, num_blocks, activation, dropout_ratio)
        self._validate_inputs()
        self.build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_dim > 0, "The number of input features should be positive."
        assert self.num_layers > 0, "The number of layers should be positive."
        assert self.num_units > 0, "The number of units should be positive."
        super(ImplicitDenseNetKernel, self)._validate_inputs()
    
    def build_layers(self) -> None:
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        if self.num_blocks == 0:
            self.dense_blocks.append(nn.Linear(self.input_dim, self.latent_feature_dim))
        else:
            self.dense_blocks.append(
                nn.Sequential(
                    nn.Linear(self.input_dim, self.num_units),
                    ACTIVATIONS[self.activation],
                    nn.Dropout(self.dropout_ratio)
                )
            )
            for _ in range(self.num_blocks - 1):
                self.dense_blocks.append(
                    nn.Sequential(
                        *[nn.Linear(self.num_units, self.num_units) for _ in range(self.num_layers)],
                        ACTIVATIONS[self.activation],
                        nn.Dropout(self.dropout_ratio)
                    )
                )
            self.dense_blocks.append(nn.Linear(self.num_units, self.latent_feature_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DenseNet
        """
        for dense_block in self.dense_blocks:
            x = dense_block(x)
        return x
    
    def get_depth(self) -> int:
        """
        Get the depth of the DenseNet
        """
        return (self.num_blocks - 1) * self.num_layers + 2

class ImplicitConvNet2DKernel(ImplicitNNKernel):
    """
    Implicit kernel implied by a 2D convolutional neural network

    Arguments
    --------------
    input_width: int, the width of input tensor
    input_height: int, the height of input tensor
    in_channels: int, the number of channels of input tensor
    num_intermediate_channels: int, the number of channels in each HIDDEN convolutional layer
    kernel_size: int, the kernel size of each convolutional layer
    stride: int, the stride of each convolutional layer
    use_batch_norm: bool, whether to use batch normalization in each convolutional block
    adaptive_avgpool_size: int, the output size of the adaptive average pooling layer
    num_hidden_dense_layers: int, the number of HIDDEN dense layers after the adaptive average pooling layer
    num_dense_units: int, the number of units in each HIDDEN dense layer
    """
    def __init__(self, input_width: int, input_height: int, in_channels: int, latent_feature_dim: int, num_blocks: int, 
                 num_intermediate_channels: int = 64, kernel_size: int = 3, stride: int = 1, use_batch_norm: bool = False, 
                 activation: str = 'relu', adaptive_avgpool_size: int = 7, num_hidden_dense_layers: int = 0, 
                 num_dense_units: int = 256, dropout_ratio: float = 0.0) -> None:
        self.input_width: int = input_width
        self.input_height: int = input_height
        self.in_channels = in_channels
        self.num_intermediate_channels = num_intermediate_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.adaptive_avgpool_size = adaptive_avgpool_size
        self.num_hidden_dense_layers = num_hidden_dense_layers
        self.num_dense_units = num_dense_units
        super(ImplicitConvNet2DKernel, self).__init__(latent_feature_dim, num_blocks, activation, dropout_ratio)
        self._validate_inputs()
        self.build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_width > 0, "The width of input features should be positive."
        assert self.input_height > 0, "The height of input features should be positive."
        assert self.in_channels > 0, "The number of input channels should be positive."
        assert self.num_intermediate_channels > 0, "The number of intermediate channels should be positive."
        assert self.kernel_size > 0, "The kernel size should be positive."
        assert self.stride > 0, "The stride should be positive."
        assert self.adaptive_avgpool_size > 0, "The adaptive average pool size should be positive."
        assert self.num_hidden_dense_layers >= 0, "The number of hidden dense layers should be non-negative."
        assert self.num_dense_units > 0, "The number of dense units should be positive."
        super(ImplicitConvNet2DKernel, self)._validate_inputs()
    
    def _build_conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, 
                          use_batch_norm: bool, activation: str) -> nn.Sequential:
        """
        Build a convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            ACTIVATIONS[activation]
        )
    
    def _build_dense_block(self, input_dim: int, output_dim: int, activation: str, dropout_ratio: float) -> nn.Sequential:
        """
        Build a dense block
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ACTIVATIONS[activation],
            nn.Dropout(dropout_ratio)
        )
    
    def build_layers(self) -> None:
        self.conv_blocks: nn.ModuleList = nn.ModuleList()
        block_expansion = self.num_intermediate_channels*self.adaptive_avgpool_size**2 if self.num_blocks > 0 else self.in_channels*self.input_width*self.input_height
        if self.num_blocks > 0:
            self.conv_blocks.append(
                self._build_conv_block(self.in_channels, self.num_intermediate_channels, self.kernel_size, self.stride,
                                    self.use_batch_norm, self.activation)
            )
            for _ in range(self.num_blocks - 1):
                self.conv_blocks.append(
                    self._build_conv_block(self.num_intermediate_channels, self.num_intermediate_channels, self.kernel_size, 
                                        self.stride, self.use_batch_norm, self.activation)
                )
            self.conv_blocks.append(nn.AdaptiveAvgPool2d(self.adaptive_avgpool_size))
        self.conv_blocks.append(nn.Flatten())
        if self.num_hidden_dense_layers == 0:    
            self.conv_blocks.append(
                nn.Linear(block_expansion, self.latent_feature_dim)
            )
        else:
            self.conv_blocks.append(
                self._build_dense_block(block_expansion, self.num_dense_units, self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_hidden_dense_layers - 1):
                self.conv_blocks.append(
                    self._build_dense_block(self.num_dense_units, self.num_dense_units, self.activation, self.dropout_ratio)
                )
            self.conv_blocks.append(
                nn.Linear(self.num_dense_units, self.latent_feature_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 2D ConvNet
        """
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
    
    def get_depth(self) -> int:
        """
        Get the depth of the 2D ConvNet
        """
        return self.num_blocks + self.num_hidden_dense_layers + 1

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