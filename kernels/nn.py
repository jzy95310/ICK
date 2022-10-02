# nn.py: a file containing the definition of implicit kernel implied by neural networks
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
import torch
from torch import nn
import vit_pytorch as vitorch
from abc import ABC, abstractmethod
from .constants import ACTIVATIONS

class ImplicitNNKernel(nn.Module, ABC):
    """
    Parent class of the implicit kernel implied by neural networks

    Arguments
    --------------
    latent_feature_dim: int, the dimension of feature in the latent space
    num_blocks: int, the number of HIDDEN blocks in the neural network, where depth = num_blocks * num_layers
    activation: str, the activation function to be used in each layer, default to 'relu'
    dropout_ratio: float, the dropout ratio for dropout layers, default to 0.0
    """
    @abstractmethod
    def __init__(self, latent_feature_dim: int, num_blocks: int, activation: str = 'relu', dropout_ratio: float = 0.0) -> None:
        super(ImplicitNNKernel, self).__init__()
        self.latent_feature_dim: int = latent_feature_dim
        self.num_blocks: int = num_blocks
        self.activation: str = activation
        self.dropout_ratio: float = dropout_ratio
        self._validate_inputs()
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the NN-implied implicit kernel class
        """
        assert self.latent_feature_dim > 0, "The number of output features should be positive."
        assert self.num_blocks >= 0, "The number of blocks should be non-negative."
        assert self.activation in ACTIVATIONS, "The activation function should be one of the following: {}".format(ACTIVATIONS.keys())
        assert 0.0 <= self.dropout_ratio <= 1.0, "The dropout ratio should be between 0.0 and 1.0."
    
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
    
    @abstractmethod
    def _build_layers(self) -> None:
        """
        Build layers of the NN-implied implicit kernel class
        """
        pass
    
    @abstractmethod
    def forward(self) -> None:
        """
        Forward pass of the NN-implied implicit kernel class
        """
        pass

    def get_depth(self) -> None:
        """
        Get the depth of the neural network
        """
        pass

class ImplicitDenseNetKernel(ImplicitNNKernel):
    """
    Implicit kernel implied by a dense neural network

    Arguments
    --------------
    input_dim: int, the dimension of input features
    num_layers_per_block: int, the number of dense layers in each HIDDEN block
    num_units: int, the number of units (width) in each HIDDEN dense layer
    """
    def __init__(self, input_dim: int, latent_feature_dim: int, num_blocks: int, num_layers_per_block: int, 
                 num_units: int, activation: str = 'relu', dropout_ratio: float = 0.0) -> None:
        self.input_dim = input_dim
        self.num_layers_per_block: int = num_layers_per_block
        self.num_units: int = num_units
        super(ImplicitDenseNetKernel, self).__init__(latent_feature_dim, num_blocks, activation, dropout_ratio)

        self.depth = (self.num_blocks - 1) * self.num_layers_per_block + 2
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_dim > 0, "The number of input features should be positive."
        assert self.num_layers_per_block > 0, "The number of layers per block should be positive."
        assert self.num_units > 0, "The number of units should be positive."
        super(ImplicitDenseNetKernel, self)._validate_inputs()
    
    def _build_layers(self) -> None:
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        if self.num_blocks == 0:
            self.dense_blocks.append(nn.Linear(self.input_dim, self.latent_feature_dim))
        else:
            self.dense_blocks.append(
                self._build_dense_block(self.input_dim, self.num_units, self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_blocks - 1):
                self.dense_blocks.append(
                    nn.Sequential(
                        *[nn.Linear(self.num_units, self.num_units) for _ in range(self.num_layers_per_block)],
                        ACTIVATIONS[self.activation],
                        nn.Dropout(self.dropout_ratio)
                    )
                )
            self.dense_blocks.append(nn.Linear(self.num_units, self.latent_feature_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dense_block in self.dense_blocks:
            x = dense_block(x)
        return x
    
    def get_depth(self) -> int:
        return self.depth
    
    def freeze_blocks(self, num_blocks_to_freeze: int) -> None:
        """
        Freeze the first num_blocks_to_freeze blocks of the DenseNet
        """
        if num_blocks_to_freeze > len(self.dense_blocks):
            raise ValueError("The number of blocks to freeze should be smaller than or equal to the number of blocks in the DenseNet.")
        for i in range(num_blocks_to_freeze):
            for param in self.dense_blocks[i].parameters():
                param.requires_grad = False
    
    def unfreeze_blocks(self, num_blocks_to_unfreeze: int) -> None:
        """
        Unfreeze the first num_blocks_to_unfreeze blocks of the DenseNet
        """
        if num_blocks_to_unfreeze > len(self.dense_blocks):
            raise ValueError("The number of blocks to unfreeze should be smaller than or equal to the number of blocks in the DenseNet.")
        for i in range(num_blocks_to_unfreeze):
            for param in self.dense_blocks[i].parameters():
                param.requires_grad = True
    
    def freeze_all_blocks(self) -> None:
        """
        Freeze all blocks in the DenseNet
        """
        self.freeze_blocks(len(self.dense_blocks))
    
    def unfreeze_all_blocks(self) -> None:
        """
        Unfreeze all blocks in the DenseNet
        """
        self.unfreeze_blocks(len(self.dense_blocks))
    
    def freeze_all_blocks_except_last(self) -> None:
        """
        Freeze all blocks except the last one of the DenseNet
        """
        self.freeze_blocks(len(self.dense_blocks) - 1)
    
    def reset_parameters_normal(self, w_mean: float = 0.0, w_std: float = 1.0, b_mean: float = 0.0, b_std: float = 1.0) -> None:
        """
        Reset the parameters of the DenseNet with a normal distribution based on NTK parameterization
        """
        for dense_block in self.dense_blocks:
            if isinstance(dense_block, nn.Linear):
                nn.init.normal_(dense_block.weight, mean=w_mean, std=w_std/np.sqrt(dense_block.out_features))
                nn.init.normal_(dense_block.bias, mean=b_mean, std=b_std)
            else:
                for layer in dense_block:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=w_mean, std=w_std/np.sqrt(layer.out_features))
                        nn.init.normal_(layer.bias, mean=b_mean, std=b_std)

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
                 activation: str = 'relu', adaptive_avgpool_size: int = 7, num_hidden_dense_layers: int = 2, 
                 num_dense_units: int = 512, dropout_ratio: float = 0.0) -> None:
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

        self.depth = self.num_blocks + self.num_hidden_dense_layers + 1
        self._validate_inputs()
        self._build_layers()
    
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
    
    def _build_layers(self) -> None:
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
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
    
    def get_depth(self) -> int:
        return self.depth
    
    def freeze_blocks(self, num_blocks_to_freeze: int) -> None:
        """
        Freeze the first num_blocks_to_freeze blocks of the 2D ConvNet
        """
        if num_blocks_to_freeze > len(self.conv_blocks):
            raise ValueError("The number of blocks to freeze should be smaller than or equal to the number of blocks in the 2D ConvNet.")
        for i in range(num_blocks_to_freeze):
            for param in self.conv_blocks[i].parameters():
                param.requires_grad = False
    
    def unfreeze_blocks(self, num_blocks_to_unfreeze: int) -> None:
        """
        Unfreeze the first num_blocks_to_unfreeze blocks of the 2D ConvNet
        """
        if num_blocks_to_unfreeze > len(self.conv_blocks):
            raise ValueError("The number of blocks to unfreeze should be smaller than or equal to the number of blocks in the 2D ConvNet.")
        for i in range(num_blocks_to_unfreeze):
            for param in self.conv_blocks[i].parameters():
                param.requires_grad = True
    
    def freeze_all_blocks(self) -> None:
        """
        Freeze all blocks in the 2D ConvNet
        """
        self.freeze_blocks(len(self.conv_blocks))
    
    def unfreeze_all_blocks(self) -> None:
        """
        Unfreeze all blocks in the 2D ConvNet
        """
        self.unfreeze_blocks(len(self.conv_blocks))
    
    def freeze_all_blocks_except_last(self) -> None:
        """
        Freeze all blocks except the last one of the DenseNet
        """
        self.freeze_blocks(len(self.conv_blocks) - 1)
    
    def freeze_all_conv_blocks(self) -> None:
        """
        Freeze all convolutional blocks of the 2D ConvNet
        """
        self.freeze_blocks(self.num_blocks)

class ImplicitViTKernel(ImplicitNNKernel):
    """
    Implicit kernel implied by a Vision Transformer (ViT)

    Arguments
    --------------
    input_width: int, the width of input tensor
    input_height: int, the height of input tensor
    patch_size: int, the size of one image patch. The image will be divided into patches of size (patch_size, patch_size).
        Must be divisible by max(input_width, input_height).
    latent_dim: int, the dimension of feature in the latent space
    num_blocks: int, the number of Transformer blocks
    readout_layer_dim: int, the dimension of the readout layer, default to 1000
    last_layer_dim: int, the dimension of the last linear layer, default to 1024
    heads: int, the number of heads in the multi-head attention layer, default to 16
    mlp_dim: int, the dimension of the MLP layer, default to 2048
    activation: str, the activation function to be used in each layer, default to 'relu'
    dropout_ratio: float, the dropout ratio, default to 0.0
    emb_dropout_ratio: float, the dropout ratio for the embedding layer, default to 0.1
    num_hidden_dense_layers: int, the number of HIDDEN dense layers after the final readout layer of ViT, default to 2
    num_dense_units: int, the number of units in each HIDDEN dense layer, default to 512
    """
    def __init__(self, input_width: int, input_height: int, patch_size: int, latent_feature_dim: int, num_blocks: int, 
                 readout_layer_dim: int = 1000, last_linear_dim: int = 1024, heads: int = 16, mlp_dim: int = 2048, 
                 activation: str = 'relu', dropout_ratio: float = 0.0, emb_dropout_ratio: float = 0.1, 
                 num_upper_hidden_dense_layers: int = 2, num_upper_dense_units: int = 512) -> None:
        self.input_size = max(input_width, input_height)
        self.patch_size = patch_size
        self.readout_layer_dim = readout_layer_dim
        self.last_linear_dim = last_linear_dim
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.emb_dropout_ratio = emb_dropout_ratio
        self.num_upper_hidden_dense_layers = num_upper_hidden_dense_layers
        self.num_upper_dense_units = num_upper_dense_units
        super(ImplicitViTKernel, self).__init__(latent_feature_dim, num_blocks, activation, dropout_ratio)
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.input_size > 0, "The input size must be positive."
        assert self.patch_size > 0, "The patch size must be positive."
        assert self.input_size % self.patch_size == 0, "The image size must be divisible by the patch size."
        assert self.readout_layer_dim > 0, "The dimension of the readout layer must be positive."
        assert self.last_linear_dim > 0, "The dimension of the last linear layer must be positive."
        assert self.heads > 0, "The number of heads in the multi-head attention layer must be positive."
        assert self.mlp_dim > 0, "The dimension of MLP layers must be positive."
        assert 0 <= self.emb_dropout_ratio < 1, "The dropout ratio for the embedding layer must be in the range [0, 1)."
        super(ImplicitViTKernel, self)._validate_inputs()
    
    def _build_vit(self) -> None:
        """
        Build the Vision Transformer (ViT) model
        """
        self.vit = vitorch.ViT(
            image_size=self.input_size, 
            patch_size=self.patch_size, 
            num_classes=self.readout_layer_dim, 
            dim=self.last_linear_dim, 
            depth=self.num_blocks, 
            heads=self.heads, 
            mlp_dim=self.mlp_dim, 
            dropout=self.dropout_ratio, 
            emb_dropout=self.emb_dropout_ratio
        )
    
    def _build_layers(self) -> None:
        self._build_vit()
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        if self.num_upper_hidden_dense_layers == 0:
            self.dense_blocks.append(
                nn.Linear(self.readout_layer_dim, self.latent_feature_dim)
            )
        else:
            self.dense_blocks.append(
                self._build_dense_block(self.readout_layer_dim, self.num_upper_dense_units, 
                                        self.activation, self.dropout_ratio)
            )
            for _ in range(self.num_upper_hidden_dense_layers - 1):
                self.dense_blocks.append(
                    self._build_dense_block(self.num_upper_dense_units, self.num_upper_dense_units, 
                                            self.activation, self.dropout_ratio)
                )
            self.dense_blocks.append(
                nn.Linear(self.num_upper_dense_units, self.latent_feature_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        for dense_block in self.dense_blocks:
            x = dense_block(x)
        return x

class ImplicitDeepViTKernel(ImplicitViTKernel):
    """
    Implicit kernel implied by a Deep Vision Transformer (DeepViT)
    """
    def __init__(self, input_width: int, input_height: int, patch_size: int, latent_feature_dim: int, num_blocks: int, 
                 readout_layer_dim: int = 1000, last_linear_dim: int = 1024, heads: int = 16, mlp_dim: int = 2048, 
                 activation: str = 'relu', dropout_ratio: float = 0.0, emb_dropout_ratio: float = 0.1, 
                 num_upper_hidden_dense_layers: int = 2, num_upper_dense_units: int = 512) -> None:
        super(ImplicitDeepViTKernel, self).__init__(input_width, input_height, patch_size, latent_feature_dim, num_blocks, 
                                                    readout_layer_dim, last_linear_dim, heads, mlp_dim, activation, dropout_ratio, 
                                                    emb_dropout_ratio, num_upper_hidden_dense_layers, num_upper_dense_units)
    
    def _build_vit(self) -> None:
        self.vit = vitorch.deepvit.DeepViT(
            image_size=self.input_size, 
            patch_size=self.patch_size, 
            num_classes=self.readout_layer_dim, 
            dim=self.last_linear_dim, 
            depth=self.num_blocks, 
            heads=self.heads, 
            mlp_dim=self.mlp_dim, 
            dropout=self.dropout_ratio, 
            emb_dropout=self.emb_dropout_ratio
        )

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