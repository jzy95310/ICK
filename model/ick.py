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