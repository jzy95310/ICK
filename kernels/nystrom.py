# nystrom.py: a file containing the definition of implicit kernel with Nystrom approximation
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import inspect
from typing import Callable, List, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
from torch import nn

class ImplicitKernel(nn.Module, ABC):
    """
    Parent class of the implicit kernel

    Arguments
    --------------
    kernel_func: Callable, a function that specifies the kernel to be computed. The kernel function takes in 
        two data points x and y and returns the computed kernel K(x,y).
    params: List[str], a list specifying the name of each parameter
    vals: List[int], a list specifying the value of each parameter
    trainable: List[bool], a list specifying whether each parameter is trainable
    """
    @abstractmethod
    def __init__(self, kernel_func: Callable, params: List[str], vals: List[float], trainable: List[bool]) -> None:
        super(ImplicitKernel, self).__init__()
        self.kernel_func: Callable = kernel_func
        self.params: List[str] = params
        self.vals: List[float] = vals
        self.trainable: List[bool] = trainable
        self._validate_inputs()
        
        # Register both trainable and un-trainable kernel parameters
        for i in range(len(self.params)):
            if self.trainable[i]:
                if isinstance(self.vals[i], list):
                    for j in range(len(self.vals[i])):
                        setattr(self, self.params[i]+"_"+str(j+1), nn.Parameter(torch.tensor(self.vals[i][j], requires_grad=True)))
                else:
                    setattr(self, self.params[i], nn.Parameter(torch.tensor(self.vals[i], requires_grad=True)))
            else:
                if isinstance(self.vals[i], list):
                    for j in range(len(self.vals[i])):
                        setattr(self, self.params[i]+"_"+str(j+1), self.vals[i][j])
                else:
                    setattr(self, self.params[i], self.vals[i])
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the implicit kernel class
        """
        if len(self.params) != len(self.vals) or len(self.params) != len(self.trainable):
            raise ValueError("The arguments \'params\', \'vals\', and \'trainable\' should all have the same length.")
        fullargs, _, _, defaults, _, _, _ = inspect.getfullargspec(self.kernel_func)
        kwargs = fullargs[-len(defaults):]
        for i in range(len(self.params)):
            if self.params[i] not in kwargs:
                raise ValueError(self.params[i] + " is not one of the arguments for the specified kernel function.")
    
    def _get_kernel_params(self) -> List[torch.Tensor]:
        """
        Get the kernel function parameters (both trainable and untrainable) from the class attributes
        """
        kernel_params = defaultdict()
        fullargs, _, _, defaults, _, _, _ = inspect.getfullargspec(self.kernel_func)
        kwargs = fullargs[-len(defaults):]
        for kwarg in kwargs:
            if hasattr(self, kwarg):
                kernel_params[kwarg] = getattr(self, kwarg)
            elif hasattr(self, kwarg+"_1"):
                i = 1
                kwarg_list = []
                while hasattr(self, kwarg+"_"+str(i)):
                    kwarg_list.append(getattr(self, kwarg+"_"+str(i)))
                    i += 1
                kernel_params[kwarg] = kwarg_list
        return kernel_params
    
    @abstractmethod
    def forward(self) -> None:
        """
        Forward pass of the implicit kernel
        """
        pass
    
class ImplicitNystromKernel(ImplicitKernel):
    """
    Implicit kernel with Nystrom approximation as kernel-to-latent-space transformation

    Arguments
    --------------
    alpha: float, a float number added to the diagonal of the kernel matrix during fitting
    num_inducing_points: int, the number of inducing points to be used in the Nystrom approximation, should be
        equal to the dimension of the latent feature returned by the NN kernels defined in nn.py
    nys_space: List[tuple(float,float)], a list of tuples specifying the lower and upper bound of the input space
        for Nystrom approximation, where len(nys_space) = # dimensions of the input feature.
        For example, if the input feature has 2 dimensions, then nys_space should be set to:
        [(lower bound of first dim, upper bound of first dim), (lower bound of second dim, upper bound of second dim)]
    """
    def __init__(self, kernel_func: Callable, params: List[str], vals: List[float], trainable: List[bool], alpha: float = 1e-5, 
                 num_inducing_points: int = 16, nys_space: List[Tuple[float]] = [(0.0,1.0)]):
        self.alpha: float = alpha
        self.num_inducing_points: int = num_inducing_points
        self.nys_space: List[Tuple(float)] = nys_space
        self.features: torch.Tensor = None
        super(ImplicitNystromKernel, self).__init__(kernel_func, params, vals, trainable)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the implicit kernel class with Nystrom approximation
        """
        super(ImplicitNystromKernel, self)._validate_inputs()
        assert self.num_inducing_points > 0, "The number of inducing points should be greater than 0."
        for i in range(len(self.nys_space)):
            assert len(self.nys_space[i]) == 2, "The input space for Nystrom approximation should be a list of tuples of ."
            assert self.nys_space[i][0] < self.nys_space[i][1], "The lower bound of the input space should be less than the upper bound."

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the implicit kernel with Nystrom approximation

        Arguments
        --------------
        input_feature: torch.Tensor, the input feature to the implicit kernel
        """
        kernel_params = self._get_kernel_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(self.nys_space) == 1:
            inducing_points = torch.linspace(self.nys_space[0][0],self.nys_space[0][1],steps=self.num_inducing_points).reshape(self.num_inducing_points,1).to(device)
        else:
            spaces = []
            for bound in self.nys_space:
                spaces.append(torch.linspace(bound[0],bound[1],steps=self.num_inducing_points).reshape(self.num_inducing_points,1).to(device))
            inducing_points = torch.cat(tuple(spaces), dim=1)
        
        # Compute the kernel matrix and map it to the latent space with Nystrom approximation and Cholesky decomposition
        Ktt = self.kernel_func(inducing_points,inducing_points,**kernel_params)  # shape: (m,m)
        Ktti = torch.inverse(Ktt+self.alpha*torch.eye(self.num_inducing_points).to(device)).double()
        Kcholt = torch.linalg.cholesky(Ktti).transpose(-2, -1).conj()
        Kxt = self.kernel_func(input_feature,inducing_points,**kernel_params).double() # shape: (N,m)
        self.features = torch.mm(Kxt,Kcholt)
        return self.features
    
    def get_features(self) -> torch.Tensor:
        """
        Get the latent representation of the input feature
        """
        return self.features

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