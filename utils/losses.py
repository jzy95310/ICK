# losses.py: a file containing the definition of PyTorch loss functions that can be useful for training ICK models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch.nn.modules.loss import _Loss

class FactualMSELoss(_Loss):
    """
    Create a criterion that measures the mean squared error (squared L2 norm) between the factual outcomes and the predicted
    outcomes, which is useful for training ICK-CMGP models for causal inference tasks.
    """
    __constants__ = ['reduction']

    def __init__(self, size_average:bool = None, reduce: bool = None, reduction: str = 'mean') -> None:
        super(FactualMSELoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between the factual outcomes and the predicted outcomes.

        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (batch_size, 2) that contains the predicted control and treatment outcomes
        target: torch.Tensor, a tensor of shape (batch_size) that contains the factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point

        Returns
        --------------
        loss: torch.Tensor, a scalar tensor that contains the mean squared error between the factual outcomes and the predicted outcomes
        """
        return torch.mean(torch.cat(((prediction[group==0,0] - target[group==0])**2, (prediction[group==1,1] - target[group==1])**2)))

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