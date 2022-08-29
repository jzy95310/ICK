# ick.py: a file containing the definition of Implicit Composite Kernel (ICK) model for regression
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import List
from kernels.nn import ImplicitNNKernel
from kernels.nystrom import ImplicitKernel

import torch
from torch import nn

class ICK(nn.Module):
    """
    Parent class of the Implicit Composite Kernel (ICK)

    Arguments
    --------------
    kernel_assignment: List, a list of ImplicitKernel or ImplicitNNKernel objects that specify the kernel and
        map the kernel into the latent space
    """
    def __init__(self, kernel_assignment: List, **kwargs):
        super(ICK, self).__init__()
        self.kernel_assignment: List = nn.ModuleList(kernel_assignment)
        self.num_modalities: int = len(self.kernel_assignment)
        self.__dict__.update(kwargs)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to ICK
        """
        assert self.num_modalities > 0, "The number of modalities (or sources of information) should be greater than 0."
        assert len(self.kernel_assignment) == self.num_modalities, "The length of the kernel assignment should be equal to num_modalities."
        for kernel in self.kernel_assignment:
            if not (isinstance(kernel, ImplicitKernel) or isinstance(kernel, ImplicitNNKernel)):
                raise ValueError("The kernel assignment should be a list of ImplicitKernel or ImplicitNNKernel objects.")
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ICK:
        Returns an inner product between latent representations
        """
        assert len(x) == self.num_modalities, "The length of the input should be equal to num_modalities."
        for i in range(len(x)):
            if 'latent_features' not in locals():
                latent_features = torch.unsqueeze(self.kernel_assignment[i](x[i]), dim=0)
            else:
                new_latent_feature = torch.unsqueeze(self.kernel_assignment[i](x[i]), dim=0)
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