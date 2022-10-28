# ick_cmgp.py: a file containing the definition of a variant of Implicit Composite Kernel (ICK) model that simulates the
# behavior of the Causal Multi-task Gaussian Process (CMGP) model in the paper "Bayesian Inference of Individualized Treatment
# Effects using Multi-task Gaussian Processes" by Ahmed M. Alaa and Mihaela van der Schaar
# arXiv: https://arxiv.org/abs/1704.02801
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from .ick import ICK
from typing import List, Tuple

import torch
from torch import nn

class ICK_CMGP(nn.Module):
    """
    Class definition of Implicit Composite Kernel-Causal Multi-task Gaussian Process (ICK-CMGP) model, which is mainly
    used for estimating the individual treatment effect (ITE) in causal inference tasks. The ITE is computed as 
    E[Y1-Y0|X] where Y0 and Y1 are the outcomes of the control and treatment groups, respectively, and X is the input. 
    Therefore, the forward pass of ICK-CMGP will return 2 outputs, one for control and one for treatment. 

    Arguments
    --------------
    control_components: List[ICK], a list of ICK objects that represents the components specifically for the control group
    treatment_components: List[ICK], a list of ICK objects that represents the components specifically for the treatment group
    shared_components: List[ICK], a list of ICK objects that represents the shared components between the control and treatment groups
    control_coeffs: List[float], a list of coefficients for the control components, default to [1.0] * len(control_components)
    treatment_coeffs: List[float], a list of coefficients for the treatment components, default to [1.0] * len(treatment_components)
    shared_coeffs: List[float], a list of coefficients for the shared components, default to [1.0] * len(shared_components)
    """
    def __init__(self, control_components: List[ICK], treatment_components: List[ICK], shared_components: List[ICK], 
                 control_coeffs: List[float] = None, treatment_coeffs: List[float] = None, shared_coeffs: List[float] = None) -> None:
        super(ICK_CMGP, self).__init__()
        self.control_components: nn.ModuleList = nn.ModuleList(control_components)
        self.treatment_components: nn.ModuleList = nn.ModuleList(treatment_components)
        self.shared_components: nn.ModuleList = nn.ModuleList(shared_components)
        self.control_coeffs: List[float] = [1.0] * len(control_components) if control_coeffs is None else control_coeffs
        self.treatment_coeffs: List[float] = [1.0] * len(treatment_components) if treatment_coeffs is None else treatment_coeffs
        self.shared_coeffs: List[float] = [1.0] * len(shared_components) if shared_coeffs is None else shared_coeffs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to ICK-CMGP
        """
        assert all([isinstance(x, ICK) for x in self.control_components]), "control_components must be a list of ICK objects."
        assert all([isinstance(x, ICK) for x in self.treatment_components]), "treatment_components must be a list of ICK objects."
        assert all([isinstance(x, ICK) for x in self.shared_components]), "shared_components must be a list of ICK objects."
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ICK-CMGP:
        Return the predicted control and treatment outcomes given the input x.
        For example, let Y0 = control outcome, Y1 = treatment outcome
        Suppose we have 1 control component f1, 1 treatment component f2, 1 shared component f3, and 
        corresponding coefficients a1, a2, a3, respectively. Then, the forward pass will return an output tensor of 
        shape (batch_size, 2) where Y0 = output[:,0] and Y1 = output[:,1], and Y0 and Y1 are computed as:
        Y0 = a1 * f1(x) + a3 * f3(x)
        Y1 = a2 * f2(x) + a3 * f3(x)
        """
        control_output = sum([a * f(x) for a, f in zip(self.control_coeffs, self.control_components)])
        treatment_output = sum([a * f(x) for a, f in zip(self.treatment_coeffs, self.treatment_components)])
        shared_output = sum([a * f(x) for a, f in zip(self.shared_coeffs, self.shared_components)])
        return torch.stack((control_output + shared_output, treatment_output + shared_output), dim=1)

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