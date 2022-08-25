# rff.py: a file containing the definition of implicit kernel with Random Fourier Features
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from .nystrom import ImplicitKernel
from typing import Callable, List
import numpy as np

import torch
from torch.fft import fft, fftshift
from torch.nn.functional import relu, gumbel_softmax

class ImplicitRFFKernel(ImplicitKernel):
    """
    Implicit kernel with Random Fourier Features as kernel-to-latent-space transformation

    Arguments
    --------------
    latent_feature_dim: int, dimension of latent space feature returned by RFF, should be equal to HALF of the 
        dimension of the latent feature returned by the NN kernels defined in nn.py
    fourier_range: int, the symmetric range of Fourier integral, should be big enough to cover the whole Fourier spectrum
    Fs: int, sampling rate [Hz] for the Discrete Fourier Transform
    """
    def __init__(self, kernel_func: Callable, params: List[str], vals: List[float], trainable: List[bool], 
                 latent_feature_dim: int = 16, fourier_range: int = 100, Fs: int = 50):
        self.latent_feature_dim = latent_feature_dim
        self.fourier_range = fourier_range
        self.Fs = Fs
        super(ImplicitRFFKernel, self).__init__(kernel_func, params, vals, trainable)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the implicit kernel class with Random Fourier Features
        """
        super(ImplicitRFFKernel, self)._validate_inputs()
        assert self.latent_feature_dim > 0, "The dimension of the latent space feature should be greater than 0."
        assert self.fourier_range > 0, "The range of the Fourier integral should be greater than 0."
        assert self.Fs > 0, "The sampling rate of DFT should be greater than 0."
    
    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the implicit kernel with Random Fourier Features

        Arguments
        --------------
        input_feature: torch.Tensor, the input feature to the implicit kernel
        """
        kernel_params = self._get_kernel_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dist = torch.arange(-self.fourier_range, self.fourier_range, 1./self.Fs).to(device)    # Range of Fourier integral   
        w = torch.tensor(np.linspace(-self.Fs*np.pi, self.Fs*np.pi, len(dist), endpoint=False)).to(device)
        # Approximate the Fourier transform of kernel_func with DFT
        ft_approx = relu(torch.real(fftshift(fft(self.kernel_func(dist,**kernel_params)) * torch.exp(1j*w*self.fourier_range) * 1/self.Fs/2/np.pi)))
        # Apply Gumbel-Softmax trick to allow auto-differentiation
        one_hot_samples = gumbel_softmax(logits=torch.log(ft_approx).reshape(1,-1).repeat(self.latent_feature_dim,1), tau=1e-8, hard=True)
        w_samples = torch.squeeze(torch.matmul(one_hot_samples, w.reshape(-1,1))).to(device)
        # Construct Random Fourier Features
        x = input_feature.reshape(1,-1).repeat(self.latent_feature_dim,1).T
        cos_features = torch.cos(w_samples*x).to(device)
        sin_features = torch.cos(w_samples*x).to(device)
        rff_features = torch.cat((cos_features,sin_features), dim=1)
        features = (torch.sqrt(torch.tensor(1/self.latent_feature_dim))*rff_features).to(device)
        return features

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