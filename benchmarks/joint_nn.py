# joint_nn.py: a file containing the definition of joint deep network benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn
from abc import ABC, abstractmethod
from kernels.constants import *

class JointNN(nn.Module, ABC):
    """
    Parent class of the joint deep network models

    Arguments
    --------------
    num_blocks: int, the number of HIDDEN blocks in the neural network backbone
    activation: str, the activation function to be used in each layer, default to 'relu'
    dropout_ratio: float, the dropout ratio for dropout layers, default to 0.0
    """
    @abstractmethod
    def __init__(self, num_blocks: int, activation: str = 'relu', dropout_ratio: float = 0.0) -> None:
        super(JointNN, self).__init__()
        self.num_blocks = num_blocks
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self._validate_inputs()
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the JointNN class
        """
        assert self.num_blocks >= 0, "The number of blocks should be non-negative."
        assert self.activation in ACTIVATIONS, "The activation function should be one of the following: {}".format(ACTIVATIONS.keys())
        assert 0.0 <= self.dropout_ratio <= 1.0, "The dropout ratio should be between 0.0 and 1.0."
    
    def _build_dense_block(self, input_dim: int, output_dim: int, activation: str, dropout_ratio: float) -> nn.Sequential:
        """
        Build a dense block
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ACTIVATIONS[activation],
            nn.Dropout(dropout_ratio)
        )
    
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
    
    @abstractmethod
    def _build_layers(self) -> None:
        """
        Build the layers of the JointNN class
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """
        Forward pass of the JointNN class
        """
        pass

class JointDenseNet(JointNN):
    """
    Benchmark model: DenseNet joint with another ML model (e.g. random forest, SVM, etc.)
    The feature returned by the ML model is passed in through the `aug_feature` argument in the forward pass
    and its dimension should be specified by the `aug_feature_dim` argument. The aug_feature will be concatenated
    to the latent feature from the dense network.

    Arguments
    --------------
    input_dim: int, the dimension of input features
    num_blocks: int, the number of HIDDEN blocks in the backbone DenseNet
    num_layers_per_block: int, the number of dense layers in each HIDDEN block
    num_units: int, the number of units (width) in each HIDDEN dense layer
    aug_feature_dim: int, dimension of the augmented feature to be concatenated with the latent feature from 
        the dense network
    latent_feature_dim: int, the dimension of feature in the latent space, default to 1000
    activation: str, the activation function to be used in each layer, default to 'relu'
    dropout_ratio: float, the dropout ratio for dropout layers, default to 0.0
    num_upper_hidden_dense_layers: int, the number of HIDDEN dense layers after the concatenated latent feature
    num_upper_dense_units: int, the number of units (width) in each HIDDEN dense layer after the concatenated
        latent feature

    References
    --------------
    [1]. Zheng, Tongshu, et al. "Local PM2. 5 Hotspot Detector at 300 m Resolution: A Random Forest–Convolutional 
    Neural Network Joint Model Jointly Trained on Satellite Images and Meteorology." Remote Sensing 13.7 (2021): 1356.
    [2]. Jiang, Ziyang, et al. "Improving spatial variation of ground-level PM2. 5 prediction with contrastive learning 
    from satellite imagery." Science of Remote Sensing 5 (2022): 100052.
    """
    def __init__(self, input_dim: int, num_blocks: int, num_layers_per_block: int, num_units: int, aug_feature_dim: int, 
                 latent_feature_dim: int = 1000,  activation: str = 'relu', dropout_ratio: float = 0.0,
                 num_upper_hidden_dense_layers: int = 2, num_upper_dense_units: int = 512) -> None:
        self.input_dim = input_dim
        self.num_layers_per_block: int = num_layers_per_block
        self.num_units: int = num_units
        self.aug_feature_dim = aug_feature_dim
        self.latent_feature_dim = latent_feature_dim
        self.num_upper_hidden_dense_layers = num_upper_hidden_dense_layers
        self.num_upper_dense_units = num_upper_dense_units
        super(JointDenseNet, self).__init__(num_blocks, activation, dropout_ratio)
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_dim > 0, "The number of input features should be positive."
        assert self.num_layers_per_block > 0, "The number of layers per block should be positive."
        assert self.num_units > 0, "The number of units should be positive."
        assert self.aug_feature_dim > 0, "The dimension of augmented feature should be positive."
        assert self.num_upper_hidden_dense_layers >= 0, "The number of hidden dense layers after the concatenated latent \
            feature should be non-negative."
        assert self.num_upper_dense_units > 0, "The number of units in the hidden dense layers after the concatenated latent \
            feature should be positive."
        super(JointDenseNet, self)._validate_inputs()
    
    def _build_dense_block(self, input_dim: int, output_dim: int, activation: str, dropout_ratio: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ACTIVATIONS[activation],
            nn.Dropout(dropout_ratio)
        )
    
    def _build_layers(self) -> None:
        self.lower_dense_blocks: nn.ModuleList = nn.ModuleList()
        self.upper_dense_blocks: nn.ModuleList = nn.ModuleList()
        if self.num_blocks == 0:
            self.lower_dense_blocks.append(nn.Linear(self.input_dim, self.latent_feature_dim + self.aug_feature_dim))
        else:
            self.lower_dense_blocks.append(
                self._build_dense_block(self.input_dim, self.num_units, self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_blocks - 1):
                self.lower_dense_blocks.append(
                    nn.Sequential(
                        *[nn.Linear(self.num_units, self.num_units) for _ in range(self.num_layers_per_block)],
                        ACTIVATIONS[self.activation],
                        nn.Dropout(self.dropout_ratio)
                    )
                )
            self.lower_dense_blocks.append(nn.Linear(self.num_units, self.latent_feature_dim + self.aug_feature_dim))
        if self.num_upper_hidden_dense_layers == 0:
            self.upper_dense_blocks.append(nn.Linear(self.latent_feature_dim + self.aug_feature_dim, 1))
        else:
            self.upper_dense_blocks.append(
                self._build_dense_block(self.latent_feature_dim + self.aug_feature_dim, self.num_upper_dense_units,
                                        self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_upper_hidden_dense_layers - 1):
                self.upper_dense_blocks.append(
                    self._build_dense_block(self.num_upper_dense_units, self.num_upper_dense_units, self.activation,
                                            self.dropout_ratio)
                )
            self.upper_dense_blocks.append(nn.Linear(self.num_upper_dense_units, 1))
    
    def forward(self, x: torch.Tensor, aug_feature: torch.Tensor) -> torch.Tensor:
        for dense_block in self.lower_dense_blocks:
            x = dense_block(x)
        x = torch.cat((x, aug_feature), dim=-1).float()
        for dense_block in self.upper_dense_blocks:
            x = dense_block(x)
        return x

class JointConvNet(JointNN):
    """
    Benchmark model: ConvNet joint with other ML models (e.g. random forest, SVM, etc.)
    The feature returned by the ML model is passed in through the `aug_feature` argument in the forward pass
    and its dimension should be specified by the `aug_feature_dim` argument. The aug_feature will be concatenated
    to the latent feature from the convolutional network.

    Arguments
    --------------
    input_width: int, the width of input tensor
    input_height: int, the height of input tensor
    in_channels: int, the number of channels of input tensor
    num_blocks: int, the number of HIDDEN blocks in the backbone ConvNet
    aug_feature_dim: int, dimension of the augmented feature to be concatenated with the latent feature from 
        the convolutional network
    num_intermediate_channels: int, the number of channels in each HIDDEN convolutional layer
    kernel_size: int, the kernel size of each convolutional layer
    stride: int, the stride of each convolutional layer
    use_batch_norm: bool, whether to use batch normalization in each convolutional block
    activation: str, the activation function to be used in each layer, default to 'relu'
    adaptive_avgpool_size: int, the output size of the adaptive average pooling layer
    num_upper_hidden_dense_layers: int, the number of HIDDEN dense layers after the concatenated latent feature
    num_upper_dense_units: int, the number of units (width) in each HIDDEN dense layer after the concatenated
        latent feature
    dropout_ratio: float, the dropout ratio for dropout layers, default to 0.0

    References
    --------------
    [1]. Zheng, Tongshu, et al. "Local PM2. 5 Hotspot Detector at 300 m Resolution: A Random Forest–Convolutional 
    Neural Network Joint Model Jointly Trained on Satellite Images and Meteorology." Remote Sensing 13.7 (2021): 1356.
    [2]. Jiang, Ziyang, et al. "Improving spatial variation of ground-level PM2. 5 prediction with contrastive learning 
    from satellite imagery." Science of Remote Sensing 5 (2022): 100052.
    """
    def __init__(self, input_width: int, input_height: int, in_channels: int, num_blocks: int, 
                 aug_feature_dim: int, num_intermediate_channels: int = 64, kernel_size: int = 3, stride: int = 1, 
                 use_batch_norm: bool = False, activation: str = 'relu', adaptive_avgpool_size: int = 7, 
                 num_upper_hidden_dense_layers: int = 2, num_upper_dense_units: int = 512, dropout_ratio: float = 0.0) -> None:
        self.input_width = input_width
        self.input_height = input_height
        self.in_channels = in_channels
        self.aug_feature_dim = aug_feature_dim
        self.num_intermediate_channels = num_intermediate_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.adaptive_avgpool_size = adaptive_avgpool_size
        self.num_upper_hidden_dense_layers = num_upper_hidden_dense_layers
        self.num_upper_dense_units = num_upper_dense_units
        super(JointConvNet, self).__init__(num_blocks, activation, dropout_ratio)
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_width > 0, 'input_width must be positive'
        assert self.input_height > 0, 'input_height must be positive'
        assert self.in_channels > 0, 'in_channels must be positive'
        assert self.aug_feature_dim > 0, 'aug_feature_dim must be positive'
        assert self.num_intermediate_channels > 0, 'num_intermediate_channels must be positive'
        assert self.kernel_size > 0, 'kernel_size must be positive'
        assert self.stride > 0, 'stride must be positive'
        assert self.adaptive_avgpool_size > 0, 'adaptive_avgpool_size must be positive'
        assert self.num_upper_hidden_dense_layers >= 0, 'The number of hidden dense layers after the concatenated latent \
            feature should be non-negative.'
        assert self.num_upper_dense_units > 0, 'The number of units in the hidden dense layers after the concatenated latent \
            feature should be positive.'
        super(JointConvNet, self)._validate_inputs()
    
    def _build_conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, 
                          use_batch_norm: bool, activation: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            ACTIVATIONS[activation]
        )
    
    def _build_dense_block(self, input_dim: int, output_dim: int, activation: str, dropout_ratio: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ACTIVATIONS[activation],
            nn.Dropout(dropout_ratio)
        )
    
    def _build_layers(self) -> None:
        self.conv_blocks: nn.ModuleList = nn.ModuleList()
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
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
        if self.num_upper_hidden_dense_layers == 0:
            self.dense_blocks.append(
                nn.Linear(block_expansion + self.aug_feature_dim, 1)
            )
        else:
            self.dense_blocks.append(
                self._build_dense_block(block_expansion + self.aug_feature_dim, self.num_upper_dense_units, 
                                        self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_upper_hidden_dense_layers - 1):
                self.dense_blocks.append(
                    self._build_dense_block(self.num_upper_dense_units, self.num_upper_dense_units, 
                                            self.activation, self.dropout_ratio)
                )
            self.dense_blocks.append(
                nn.Linear(self.num_upper_dense_units, 1)
            )
    
    def forward(self, x: torch.Tensor, aug_feature: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = torch.cat([x, aug_feature], dim=1).float()
        for dense_block in self.dense_blocks:
            x = dense_block(x)
        return x

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