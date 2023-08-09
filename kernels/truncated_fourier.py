# fourier.py: a file containing the definition of truncated Fourier expansion of periodic functions
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
from typing import Callable
import torch
from torch import nn
from torchquad import Simpson

class TruncatedFourierExpansion(nn.Module):
    """
    Truncated Fourier expansion of periodic functions

    Arguments
    --------------
    func: Callable, the periodic function to be approximated
    T: float, the initial value of the period of the function
    num_terms: int, the number of terms in the truncated Fourier expansion
    num_int_pts: int, the number of points used in the Monte Carlo integration
    """
    def __init__(self, func: Callable, T: float, num_terms: int = 50, num_int_pts: int = 4999) -> None:
        super(TruncatedFourierExpansion, self).__init__()
        self.func: Callable = func
        self.num_terms: int = num_terms
        self.num_int_pts: int = num_int_pts

        # Register T as trainable parameter
        self.T = nn.Parameter(torch.tensor(T, requires_grad=True))

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """
        Compute the truncated Fourier expansion of the input feature
        """
        x_dim, device = input_feature.shape[0], input_feature.device
        sp = Simpson()
        A0 = 1/self.T * sp.integrate(self.func, dim=1, N=self.num_int_pts, integration_domain=[[-self.T/2,self.T/2]], backend="torch")
        An = torch.tensor([2/self.T * sp.integrate((
            lambda x: self.func(x.to(device))*torch.cos(2*np.pi*n*x.to(device)/self.T)), dim=1, N=self.num_int_pts, integration_domain=[[-self.T/2,self.T/2]], backend="torch") 
        for n in range(1,self.num_terms+1)])
        An = torch.broadcast_to(An, (x_dim,self.num_terms)).to(device)  # shape: (x_dim, num_terms)
        Bn = torch.tensor([2/self.T * sp.integrate((
            lambda x: self.func(x.to(device))*torch.sin(2*np.pi*n*x.to(device)/self.T)), dim=1, N=self.num_int_pts, integration_domain=[[-self.T/2,self.T/2]], backend="torch") 
        for n in range(1,self.num_terms+1)])  
        Bn = torch.broadcast_to(Bn, (x_dim,self.num_terms)).to(device)  # shape: (x_dim, num_terms)
        cos_series = torch.stack([torch.cos(2*np.pi*n*input_feature/self.T) for n in range(1,self.num_terms+1)]).squeeze().T  # shape: (x_dim, num_terms)
        sin_series = torch.stack([torch.sin(2*np.pi*n*input_feature/self.T) for n in range(1,self.num_terms+1)]).squeeze().T  # shape: (x_dim, num_terms)
        return (A0 + torch.sum(An*cos_series+Bn*sin_series, dim=1)).view(-1,1)    

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