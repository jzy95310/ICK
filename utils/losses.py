# losses.py: a file containing the definition of PyTorch loss functions that can be useful for training ICK/CMDE 
# models on causal inference tasks
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from geomloss import SamplesLoss

class FactualMSELoss(_Loss):
    """
    Create a criterion that measures the mean squared error (squared L2 norm) between the factual outcomes and the predicted
    outcomes, which is useful for training ICK-CMGP models for causal inference tasks.

    Arguments
    --------------
    regularize_var: bool, whether to regularize the variance of the COUNTERFACTUAL predicted outcomes, only supported for 
        CMICK ensemble model, default to False
    """
    __constants__ = ['reduction']

    def __init__(self, size_average: bool = None, reduce: bool = None, reduction: str = 'mean', regularize_var: bool = False) -> None:
        self.regularize_var = regularize_var
        super(FactualMSELoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between the factual outcomes and the predicted outcomes.

        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (num_estimators, batch_size, 2) or (batch_size, 2) that 
            contains the predicted control and treatment outcomes
        target: torch.Tensor, a tensor of shape (batch_size,) that contains the true factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point

        Returns
        --------------
        loss: torch.Tensor, a scalar tensor that contains the mean squared error between the factual outcomes and the predicted outcomes
        """
        if len(prediction.shape) >= 3:
            factual_err = F.mse_loss(torch.cat((torch.mean(prediction,dim=0)[group==0,0], torch.mean(prediction,dim=0)[group==1,1])), 
                                    torch.cat((target[group==0], target[group==1])), reduction=self.reduction)
            counterfactual_var = torch.cat((torch.var(prediction,dim=0)[group==1,0], torch.var(prediction,dim=0)[group==0,1]))
        else:
            factual_err = F.mse_loss(prediction[group==0,0], target[group==0], reduction=self.reduction) + \
                          F.mse_loss(prediction[group==1,1], target[group==1], reduction=self.reduction)
            counterfactual_var = 0.
        if self.regularize_var:
            return factual_err + torch.mean(counterfactual_var)
        else:
            return factual_err

class FactualMSELoss_MT(_Loss):
    """
    Create a criterion that measures the mean squared error (squared L2 norm) between the factual outcomes and the predicted
    outcomes with multiple treatments.
    """
    __constants__ = ['reduction']

    def __init__(self, size_average: bool = None, reduce: bool = None, reduction: str = 'mean') -> None:
        super(FactualMSELoss_MT, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between the factual outcomes and the predicted outcomes.
        """
        n_treatments = prediction.shape[-1]  # Number of treatments
        group = group.int()

        # Factual error computation
        if len(prediction.shape) >= 3:
            mean_pred = torch.mean(prediction, dim=0)  # shape (batch_size, n_treatments)
            factual_err = F.mse_loss(mean_pred[torch.arange(mean_pred.shape[0]), group], target, reduction=self.reduction)
            # Counterfactual variance
            counterfactual_var = []
            for t in range(n_treatments):
                mask = (group != t)
                if mask.any():
                    counterfactual_var.append(torch.var(prediction[:, mask, t], dim=0))
            counterfactual_var = torch.cat(counterfactual_var) if counterfactual_var else torch.tensor(0.0, device=prediction.device)
        else:
            # Single estimator case
            factual_err = F.mse_loss(prediction[torch.arange(prediction.shape[0]), group], target, reduction=self.reduction)
            # Counterfactual variance
            counterfactual_var = []
            for t in range(n_treatments):
                mask = (group != t)
                if mask.any():
                    counterfactual_var.append(torch.var(prediction[mask, t]))
            counterfactual_var = torch.stack(counterfactual_var) if counterfactual_var else torch.tensor(0.0, device=prediction.device)

        return factual_err

class FactualCrossEntropyLoss(_WeightedLoss):
    """
    Create a criterion that measures the cross entropy between the factual outcomes and the predicted outcomes, which is 
    useful for training ICK-CMGP models for causal inference tasks with binary or one-hot outcomes.

    Arguments
    --------------
    regularize_var: bool, whether to regularize the variance of the COUNTERFACTUAL predicted outcomes, only supported for 
        CMICK ensemble model, default to False
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
        target: torch.Tensor, a tensor of shape (batch_size,) that contains the true factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point

        Returns
        --------------
        loss: torch.Tensor, a scalar tensor that contains the cross entropy between the factual outcomes and the predicted outcomes
        """
        if len(prediction.shape) >= 3:
            factual_err = F.cross_entropy(torch.cat((torch.mean(prediction,dim=0)[group==0,0], torch.mean(prediction,dim=0)[group==1,1])), 
                                        torch.cat((target[group==0], target[group==1])), weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction, label_smoothing=self.label_smoothing)
            counterfactual_var = torch.cat((torch.var(prediction,dim=0)[group==1,0], torch.var(prediction,dim=0)[group==0,1]))
        else:
            factual_err = F.cross_entropy(prediction[group==0,0], target[group==0], weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction, label_smoothing=self.label_smoothing) + \
                          F.cross_entropy(prediction[group==1,1], target[group==1], weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction, label_smoothing=self.label_smoothing)
            counterfactual_var = 0.
        if self.regularize_var:
            return factual_err + torch.mean(counterfactual_var)
        else:
            return factual_err

class CFRLoss(_Loss):
    """
    Counterfactual regression loss as proposed by Shalit et al. (2017)

    Arguments
    --------------
    alpha: float, regularization hyperparameter for integral probability metric (IPM), default to 1e-3
    """
    ipm_metric = {
        'W1': SamplesLoss(loss="sinkhorn", p=1, backend='tensorized'), 
        'W2': SamplesLoss(loss="sinkhorn", p=2, backend='tensorized'), 
        'MMD': SamplesLoss(loss="energy", backend='tensorized')
    }

    def __init__(self, size_average: bool = None, reduce: bool = None, reduction: str = 'mean', 
                 alpha: float = 1e-3, metric: str = 'W2') -> None:
        assert metric in self.ipm_metric.keys(), "The metric must be one of the following: {}".format(self.ipm_metric.keys())
        self.alpha = alpha
        self.metric = metric
        super(CFRLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor, 
                phi_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the counterfactual regression loss between the factual outcomes and the predicted outcomes

        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (batch_size,) that contains the predicted outcomes 
            only for factual observations
        target: torch.Tensor, a tensor of shape (batch_size,) that contains the true factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the group assignment of each data point
        phi_output: torch.Tensor, a tensor of shape (batch_size, phi_width) that contains the output
            from the representation network Phi
        """
        weight = group/(2*torch.mean(group)) + (1-group)/(2*(1-torch.mean(group)))
        phi0, phi1 = phi_output[group==0], phi_output[group==1]
        factual_err = torch.mean(weight*(target-prediction)**2)
        imbalance_term = self.ipm_metric[self.metric](phi0, phi1)
        return factual_err + self.alpha*imbalance_term

class DONUTLoss(_Loss):
    """
    Factual loss with orthogonal regularization used for DONUT model as proposed by Hatt and 
    Stefan (2021). The loss is composed of an MSE term for observed outcomes, a cross-entropy 
    term for observed group assignments, and an orthogonal regularization term which consists 
    of a pseudo outcome and a perturbation function

    Arguments
    --------------
    alpha: float, weight for cross-entropy term
    beta: float, weight for orthogonal regularization term
    """
    def __init__(self, size_average: bool = None, reduce: bool = None, reduction: str = 'mean', 
                 alpha: float = 1., beta: float = 1.) -> None:
        self.alpha = alpha
        self.beta = beta
        super(DONUTLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor, 
                global_pred: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        --------------
        prediction: torch.Tensor, a tensor of shape (batch_size, 4) that is a concatenated tensor of 
            the predicted outcomes for control and treatment groups, the predicted group assignments,
            and the model parameter epsilon
        target: torch.Tensor, a tensor of shape (batch_size,) that contains the true factual outcomes
        group: torch.Tensor, a tensor of shape (batch_size,) that contains the true group assignments
        global_pred: torch.Tensor, a tensor of shape (N_train, 2) that contains the predicted potential
            outcomes for all training data points, used for estimating ATE
        """
        y0_pred, y1_pred, group_pred, epsilons = prediction[:, 0], prediction[:, 1], prediction[:, 2], prediction[:, 3]
        # Factual loss
        mse_loss = F.mse_loss(y0_pred[group==0], target[group==0], reduction=self.reduction) + \
                   F.mse_loss(y1_pred[group==1], target[group==1], reduction=self.reduction)
        t_pred = (group_pred + 0.001) / 1.002
        bce_loss = F.binary_cross_entropy(t_pred, group, reduction=self.reduction)
        # Orthogonal regularization
        t_pred = (group_pred + 0.01) / 1.02
        y0_pert = y0_pred + epsilons * (group - t_pred)
        psi = torch.mean(global_pred[:, 1] - global_pred[:, 0])
        orthogonal_reg = torch.mean((target - psi * group - y0_pert)**2)
        return mse_loss + self.alpha * bce_loss + self.beta * orthogonal_reg

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