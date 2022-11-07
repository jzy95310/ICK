# losses.py: a file containing the definition of PyTorch loss functions that can be useful for training ICK models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

class FactualMSELoss(_Loss):
    """
    Create a criterion that measures the mean squared error (squared L2 norm) between the factual outcomes and the predicted
    outcomes, which is useful for training ICK-CMGP models for causal inference tasks.

    Arguments
    --------------
    regularize_var: bool, whether to regularize the variance of the COUNTERFACTUAL predicted outcomes, only supported for 
        ICK-CMGP ensemble model, default to False
    """
    __constants__ = ['reduction']

    def __init__(self, size_average:bool = None, reduce: bool = None, reduction: str = 'mean', regularize_var: bool = False) -> None:
        self.regularize_var = regularize_var
        super(FactualMSELoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between the factual outcomes and the predicted outcomes.

        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (num_estimators, batch_size, 2) that contains the predicted 
            control and treatment outcomes for all estimators in the ensemble
        target: torch.Tensor, a tensor of shape (batch_size) that contains the factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point

        Returns
        --------------
        loss: torch.Tensor, a scalar tensor that contains the mean squared error between the factual outcomes and the predicted outcomes
        """
        factual_err = F.mse_loss(torch.cat((torch.mean(prediction,dim=0)[group==0,0], torch.mean(prediction,dim=0)[group==1,1])), 
                                 torch.cat((target[group==0], target[group==1])), reduction=self.reduction)
        counterfactual_var = torch.cat((torch.var(prediction,dim=0)[group==1,0], torch.var(prediction,dim=0)[group==0,1]))
        if self.regularize_var:
            return factual_err + torch.mean(counterfactual_var)
        else:
            return factual_err

class FactualCrossEntropyLoss(_WeightedLoss):
    """
    Create a criterion that measures the cross entropy between the factual outcomes and the predicted outcomes, which is 
    useful for training ICK-CMGP models for causal inference tasks with binary or one-hot outcomes.

    Arguments
    --------------
    regularize_var: bool, whether to regularize the variance of the COUNTERFACTUAL predicted outcomes, only supported for 
        ICK-CMGP ensemble model, default to False
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, regularize_var: bool = False) -> None:
        super(FactualCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing
        self.regularize_var: bool = regularize_var
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross entropy between the factual outcomes and the predicted outcomes.

        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (num_estimators, batch_size, 2) that contains the predicted 
            control and treatment outcomes for all estimators in the ensemble
        target: torch.Tensor, a tensor of shape (batch_size) that contains the factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point

        Returns
        --------------
        loss: torch.Tensor, a scalar tensor that contains the cross entropy between the factual outcomes and the predicted outcomes
        """
        factual_err = F.cross_entropy(torch.cat((torch.mean(prediction,dim=0)[group==0,0], torch.mean(prediction,dim=0)[group==1,1])), 
                                      torch.cat((target[group==0], target[group==1])), weight=self.weight, ignore_index=self.ignore_index,
                                      reduction=self.reduction, label_smoothing=self.label_smoothing)
        counterfactual_var = torch.cat((torch.var(prediction,dim=0)[group==1,0], torch.var(prediction,dim=0)[group==0,1]))
        if self.regularize_var:
            return factual_err + torch.mean(counterfactual_var)
        else:
            return factual_err

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