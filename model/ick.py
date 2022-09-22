# ick.py: a file containing the definition of Implicit Composite Kernel (ICK) model for regression
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import List, Dict, Union
from kernels.bnn import *
from kernels.nn import *
from kernels.nystrom import *
from kernels.rff import *

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
        """
        self.kernels = nn.ModuleList()
        for i in range(self.num_modalities):
            self.kernels.append(eval(self.kernel_assignment[i])(**self.kernel_params[self.kernel_assignment[i]]))
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ICK:
        Returns an inner product between latent representations
        """
        assert len(x) == self.num_modalities, "The length of the input should be equal to num_modalities."
        for i in range(len(x)):
            if 'latent_features' not in locals():
                latent_features = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
            else:
                new_latent_feature = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
                latent_features = torch.cat((latent_features, new_latent_feature), dim=0)
        return torch.sum(torch.prod(latent_features,dim=0),dim=1)

class BayesianICK(ICK):
    """
    Class definition of the Bayesian Implicit Composite Kernel (Bayesian ICK)
    """
    def __init__(self, kernel_assignment: List[str], kernel_params: Dict) -> None:
        super(BayesianICK, self).__init__(kernel_assignment, kernel_params)
        kernel_1_attr = list(self.kernel_params.values())[0]
        latent_feature_dim = kernel_1_attr.get('latent_feature_dim', kernel_1_attr.get('num_inducing_points', -1))
        assert latent_feature_dim > 0, "The latent feature dimension should be greater than 0."
        self.fc = nn.Linear(latent_feature_dim, 1)
        self.softplus = nn.Softplus()
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of Bayesian ICK:
        Returns a tuple of two outputs for Gaussian mean and variance where mean is the inner product between 
        latent representations and variance is directly modeled by the ImplicitNNKernel
        """
        assert len(x) == self.num_modalities, "The length of the input should be equal to num_modalities."
        for i in range(len(x)):
            if 'latent_features' not in locals():
                latent_features = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
            else:
                new_latent_feature = torch.unsqueeze(self.kernels[i](x[i]), dim=0)
                latent_features = torch.cat((latent_features, new_latent_feature), dim=0)
        mean = torch.sum(torch.prod(latent_features,dim=0),dim=1)
        for i in range(len(self.kernels)):
            if isinstance(self.kernels[i], ImplicitNNKernel):
                var = torch.squeeze(self.softplus(self.fc(latent_features[i].float())))
                break
        return mean, var if 'var' in locals() else mean

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