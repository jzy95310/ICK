# ick.py: a file containing the definition of Implicit Composite Kernel (ICK) model for regression
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import List, Dict
from kernels.bnn import *
from kernels.nn import *
from kernels.nystrom import *
from kernels.rff import *
from kernels.truncated_fourier import *
from utils.helpers import attach_single_output_dense_layer

import math
import torch
from torch import nn

class ICK(nn.Module):
    """
    Class definition of the Implicit Composite Kernel (ICK)
    Please see the notebook tutorial_1d_regression.ipynb for more details on the usage of ICK

    Arguments
    --------------
    kernel_assignment: List[str], a list of strings indicating the type of kernel to be used for each modality
    kernel_params: Dict, a dictionary containing the parameters of the kernels
    """
    def __init__(self, kernel_assignment: List[str], kernel_params: Dict) -> None:
        super(ICK, self).__init__()
        self.kernel_assignment: List[str] = kernel_assignment
        self.kernel_params: Dict = kernel_params
        self.num_modalities: int = len(self.kernel_assignment)
        self._validate_inputs()
        self._build_kernels()

    def _validate_inputs(self) -> None:
        """
        Validate the inputs to ICK
        """
        assert self.num_modalities > 0, "The number of modalities (or sources of information) should be greater than 0."
        for kernel_name in self.kernel_assignment:
            assert kernel_name in self.kernel_params.keys(), "The kernel name {} has no parameters specified.".format(kernel_name)
    
    def _build_kernels(self) -> None:
        """
        Build the kernels for each modality

        Notes
        --------------
        By default, in ICK, the latent representations are combined using an inner product mechanism. However, when there is only 
        one modality, the latent representation is simply the summation of the components of the latent representation. However,
        if the kernel is an instance of ImplicitNNKernel, then a linear layer of dimension (latent_feature_dim, 1) will be 
        attached to the final layer of the kernel.
        """
        self.kernels = nn.ModuleList()
        for i in range(self.num_modalities):
            self.kernels.append(eval(self.kernel_assignment[i])(**self.kernel_params[self.kernel_assignment[i]]))
        if self.num_modalities == 1 and isinstance(self.kernels[0], ImplicitNNKernel):
            self.kernels[0] = attach_single_output_dense_layer(self.kernels[0], self.kernels[0].activation, self.kernels[0].dropout_ratio)
        if self.num_modalities == 1 and isinstance(self.kernels[0], ImplicitNystromKernel):
            self.weights: nn.Parameter = nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty(1, self.kernels[0].num_inducing_points), a=math.sqrt(5))
            )
        if isinstance(self.kernels[0], (ImplicitNNKernel, ImplicitRFFKernel, ImplicitResNetKernel)):
            self.latent_feature_dim = self.kernels[0].latent_feature_dim
        elif isinstance(self.kernels[0], ImplicitNystromKernel):
            self.latent_feature_dim = self.kernels[0].num_inducing_points
        else:
            raise NotImplementedError("The kernel {} is not supported.".format(self.kernel_assignment[0]))
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ICK:
        Returns an inner product between latent representations
        """
        latent_features = self.get_latent_features(x)
        return torch.sum(torch.prod(latent_features,dim=0),dim=1)
    
    def get_latent_features(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Get latent features from the forward pass
        """
        assert len(x) == self.num_modalities, "The length of the input should be equal to num_modalities."
        if self.num_modalities == 1 and isinstance(self.kernels[0], ImplicitNystromKernel):
            latent_features = torch.unsqueeze(self.weights, dim=0).repeat(1, x[0].shape[0], 1)
        for i in range(len(x)):   # Here len(x) is the number of modalities
            if 'latent_features' not in locals():
                latent_features = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
            else:
                new_latent_feature = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
                latent_features = torch.cat((latent_features, new_latent_feature), dim=0)
        return latent_features
    
class AdditiveICK(nn.Module):
    """
    Class definition of the Additive Implicit Composite Kernel (ICK)
    This class is used to construct ICK which models a GP with non-separable kernel using a summation formulation

    Arguments
    --------------
    components: List[ICK], a list of ICK objects to be added together
    component_assignment: List[List[int]], a list of lists of integers indicating the assignment of each modality 
        to each ICK component. For example, let x[0] be the first modality and x[1] be the second modality, 
        if component_assignment = [[0],[1],[0,1]], then the first component will be ICK(x[0]), the second component 
        will be ICK(x[1]), and the third component will be ICK(x[0],x[1]). This is approximately equivalent to a 
        GP with non-separable kernel K = K1(x[0],x[0]') + K2(x[1],x[1]') + K3(x[0],x[0]')K3(x[1],x[1]')
    coeffs: List[float], a list of coefficients for each component. If None, all coefficients will be set to 1.0
    weighted: List[bool], a list of booleans indicating whether each component is weighted. If None, all components
        will be set to False
    """
    def __init__(self, components: List[ICK], component_assignment: List[List[int]], coeffs: List[float] = None, 
                 weighted: List[bool] = None) -> None:
        super(AdditiveICK, self).__init__()
        self.components: nn.ModuleList = nn.ModuleList(components)
        self.component_assignment: List[List[int]] = component_assignment
        self.coeffs: List[float] = [1.0] * len(self.components) if coeffs is None else coeffs
        self.weighted: List[bool] = [False] * len(self.components) if weighted is None else weighted

        self.num_components: int = len(self.components)
        self.weights: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty(1, self.components[i].latent_feature_dim), a=math.sqrt(5))
            ) for i in range(self.num_components)
        ])
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to AdditiveICK
        """
        assert self.num_components > 0, "The number of components should be greater than 0."
        assert all([isinstance(x, ICK) for x in self.components]), "All components should be ICK objects."
        assert len(self.component_assignment) == self.num_components, "The length of component_assignment should be equal to the number of components."
        assert len(self.coeffs) == self.num_components, "The length of coeffs should be equal to the number of components."
        assert len(self.weighted) == self.num_components, "The length of weighted should be equal to the number of components."
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of AdditiveICK:
        Returns a sum of the outputs from the forward pass of all ICK components
        """
        batch_size = x[0].shape[0]
        for i in range(self.num_components):
            xs = [x[j] for j in self.component_assignment[i]]
            if 'res' not in locals():
                if self.weighted[i]:
                    res = self.coeffs[i] * torch.sum(torch.prod(torch.cat(
                        (self.components[i].get_latent_features(xs), self.weights[i].repeat(1,batch_size,1))
                    ),dim=0),dim=1)
                else:
                    res = self.coeffs[i] * self.components[i](xs)
            else:
                if self.weighted[i]:
                    res += self.coeffs[i] * torch.sum(torch.prod(torch.cat(
                        (self.components[i].get_latent_features(xs), self.weights[i].repeat(1,batch_size,1))
                    ),dim=0),dim=1)
                else:
                    res += self.coeffs[i] * self.components[i](xs)
        return res

class BinaryICK(ICK):
    """
    Class definition of the Binary Implicit Composite Kernel (ICK)
    Here "binary" means that the inner product will be passed through a sigmoid function to get a probability,
    which is used for binary classification

    Arguments
    --------------
    kernel_assignment: List[str], a list of strings indicating the type of kernel to be used for each modality
    kernel_params: Dict, a dictionary containing the parameters of the kernels
    """
    def __init__(self, kernel_assignment: List[str], kernel_params: Dict) -> None:
        super(BinaryICK, self).__init__(kernel_assignment, kernel_params)
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of BinaryICK
        """
        latent_features = self.get_latent_features(x)
        return torch.sigmoid(torch.sum(torch.prod(latent_features,dim=0),dim=1))

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