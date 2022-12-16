# kernel_fn.py: a file containing the definition of kernel functions for Nystrom approximation
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
from typing import Tuple, List
import torch

###################################### Kernels for Nystrom method ######################################

def _reshape_inputs(x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Reshape the inputs to the kernel function to be compatible with the kernel function
    """
    x = x.reshape(-1, 1, x.shape[-1])
    y = y.reshape(1, -1, y.shape[-1])
    return x, y

# Periodic kernel function
def periodic_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., period:float=1., lengthscale:float=0.5, 
                        noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    diff = torch.sum(torch.abs(x-y), dim=2)
    return std*torch.exp(-2*torch.sin(np.pi*diff/period)**2/lengthscale**2) + noise*(diff==0)

# Squared Exponential kernel with noise, equivalent to Matern kernel with nu = inf
def sq_exp_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., lengthscale:float=0.5, noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    sq_diff = torch.sum((x-y)**2, dim=2)
    return std*torch.exp(-sq_diff/lengthscale**2) + noise*(sq_diff==0)

# Exponential kernel with noise, equivalent to Matern kernel with nu = 1/2
def exp_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., lengthscale:float=0.5, noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    diff = torch.sum(torch.abs(x-y), dim=2)
    return std*torch.exp(-diff/lengthscale) + noise*(diff==0)

# Rational Quadratic kernel with noise
def rq_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., lengthscale:float=0.5, scale_mixture:float=0.5, 
                  noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    sq_diff = torch.sum((x-y)**2, dim=2)
    return std*(1.+(sq_diff/(2.*scale_mixture*lengthscale**2)))**(-scale_mixture) + noise*(sq_diff==0)

# Matern kernel with noise, nu = 3/2
def matern_type1_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., lengthscale:float=0.5, noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    diff = torch.sum(torch.abs(x-y), dim=2)
    return std*(1.+torch.sqrt(torch.tensor(3.))*diff/lengthscale)*torch.exp(-torch.sqrt(torch.tensor(3.))*diff/lengthscale) + noise*(diff==0)

# Matern kernel with noise, nu = 5/2
def matern_type2_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., lengthscale:float=0.5, noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    diff = torch.sum(torch.abs(x-y), dim=2)
    sq_diff = torch.sum((x-y)**2, dim=2)
    return std*(1.+torch.sqrt(torch.tensor(5.))*diff/lengthscale+5*sq_diff/(3*lengthscale**2))*torch.exp(-torch.sqrt(torch.tensor(5.))*diff/lengthscale) + noise*(diff==0)

# Linear kernel with noise
def linear_kernel_nys(x:torch.Tensor, y:torch.Tensor, std:float=1., c:float=0., noise:float=0.5) -> torch.Tensor:
    x, y = _reshape_inputs(x, y)
    diff = torch.sum(torch.abs(x-y), dim=2)
    return std*torch.sum((x-c)*(y-c),dim=2) + noise*(diff==0)

# Spectral mixture kernel for 1D data
def spectral_mixture_kernel_1d_nys(x:torch.Tensor, y:torch.Tensor, weight:List[float]=[1.], mean:List[float]=[0.], 
                                   cov:List[float]=[0.5], noise:List[float]=0.5) -> torch.Tensor:
    assert len({len(i) for i in [weight, mean, cov]}) == 1, "All parameters must have the same length."
    x, y = _reshape_inputs(x, y)
    tau, tau_sq = torch.sum((x-y), dim=2), torch.sum((x-y)**2, dim=2)
    for i in range(len(weight)):
        if i == 0:
            mixtures = torch.unsqueeze(weight[i]*torch.exp(-2*np.pi**2*tau_sq*cov[i])*torch.cos(2*np.pi*tau*mean[i]), dim=-1)
        else:
            mixtures = torch.cat((mixtures, torch.unsqueeze(weight[i]*torch.exp(-2*np.pi**2*tau_sq*cov[i])*torch.cos(2*np.pi*tau*mean[i]), dim=-1)), dim=-1)
    return torch.sum(mixtures, dim=-1) + noise*(tau==0)

###################################### Kernels for Random Fourier Features method ######################################

# Periodic kernel function
def periodic_kernel_rff(dist_rff:torch.Tensor, std:float=1., period:float=1., lengthscale:float=.5, noise:float=0.5) -> torch.Tensor:
    return std*torch.exp(-2*torch.sin(np.pi*torch.abs(dist_rff)/period)**2/lengthscale**2) + noise*(dist_rff==0)

# Squared Exponential kernel
def sq_exp_kernel_rff(dist_rff:torch.Tensor, lengthscale:float=0.5, std:float=1, noise:float=0.5):
    return std*torch.exp(-dist_rff**2/(2*lengthscale**2)) + noise*(dist_rff==0)

# Rational Quadratic kernel
def rq_kernel_rff(dist_rff:torch.Tensor, std:float=1., lengthscale:float=.5, alpha:float=.5, noise:float=0.5):
    return std*torch.pow(1.+(dist_rff**2)/(2.*alpha*lengthscale**2), -alpha) + noise*(dist_rff==0)

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