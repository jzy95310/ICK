# cmick.py: a file containing the definition of a Causal Multi-task Implicit Composite Kernel (CMICK) model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from .ick import ICK
from typing import List

import torch
from torch import nn

class CMICK(nn.Module):
    """
    Class definition of Causal Multi-task Implicit Composite Kernel (CMICK) model, which is mainly
    used for estimating the individualized treatment effect (ITE) in causal inference tasks. The ITE is computed as 
    E[Y1-Y0|X] where Y0 and Y1 are the outcomes of the control and treatment groups, respectively, and X is the input. 
    Therefore, the forward pass of CMICK will return 2 outputs, one for control and one for treatment. 

    Arguments
    --------------
    control_components: List[ICK], a list of ICK objects that represents the components specifically for the control group
    treatment_components: List[ICK], a list of ICK objects that represents the components specifically for the treatment group
    shared_components: List[ICK], a list of ICK objects that represents the shared components between the control and treatment groups
    control_coeffs: List[float], a list of coefficients for the control components, default to [1.0] * len(control_components)
    treatment_coeffs: List[float], a list of coefficients for the treatment components, default to [1.0] * len(treatment_components)
    shared_coeffs: List[float], a list of coefficients for the shared components, default to [1.0] * len(shared_components)
    coeff_trainable: bool, whether the coefficients are trainable or not, default to False
    output_binary: bool, whether the output is binary or not, default to False. If True, the output will be passed into a 
        softmax function before returning.
    """
    def __init__(self, control_components: List[ICK], treatment_components: List[ICK], shared_components: List[ICK], 
                 control_coeffs: List[float] = None, treatment_coeffs: List[float] = None, shared_coeffs: List[float] = None, 
                 coeff_trainable: bool = False, output_binary: bool = False) -> None:
        super(CMICK, self).__init__()
        self.control_components: nn.ModuleList = nn.ModuleList(control_components)
        self.treatment_components: nn.ModuleList = nn.ModuleList(treatment_components)
        self.shared_components: nn.ModuleList = nn.ModuleList(shared_components)
        self.control_coeffs: List[float] = [1.0] * len(control_components) if control_coeffs is None else control_coeffs
        self.treatment_coeffs: List[float] = [1.0] * len(treatment_components) if treatment_coeffs is None else treatment_coeffs
        self.shared_coeffs: List[float] = [1.0] * len(shared_components) if shared_coeffs is None else shared_coeffs
        self.coeff_trainable: bool = coeff_trainable
        self.output_binary: bool = output_binary
        self._validate_inputs()
        if self.coeff_trainable:
            self._register_params()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to ICK-CMGP
        """
        assert all([isinstance(x, ICK) for x in self.control_components]), "control_components must be a list of ICK objects."
        assert all([isinstance(x, ICK) for x in self.treatment_components]), "treatment_components must be a list of ICK objects."
        assert all([isinstance(x, ICK) for x in self.shared_components]), "shared_components must be a list of ICK objects."
    
    def _register_params(self) -> None:
        """
        Register the coefficients as trainable parameters
        """
        for i in range(len(self.control_coeffs)):
            setattr(self, "control_coeff_{}".format(i+1), nn.Parameter(torch.tensor(self.control_coeffs[i], requires_grad=True)))
        for i in range(len(self.treatment_coeffs)):
            setattr(self, "treatment_coeff_{}".format(i+1), nn.Parameter(torch.tensor(self.treatment_coeffs[i], requires_grad=True)))
        for i in range(len(self.shared_coeffs)):
            setattr(self, "shared_coeff_{}".format(i+1), nn.Parameter(torch.tensor(self.shared_coeffs[i], requires_grad=True)))
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of CMICK:
        Return the predicted control and treatment outcomes given the input x.
        For example, let Y0 = control outcome, Y1 = treatment outcome
        Suppose we have 1 control component f1, 1 treatment component f2, 1 shared component f3, and 
        corresponding coefficients a1, a2, a3, respectively. Then, the forward pass will return an output tensor of 
        shape (batch_size, 2) where Y0 = output[:,0] and Y1 = output[:,1], and Y0 and Y1 are computed as:
        Y0 = a1 * f1(x) + a3 * f3(x)
        Y1 = a2 * f2(x) + a3 * f3(x)
        """
        if self.coeff_trainable:
            control_output = sum([a * f(x) for a, f in zip([getattr(self, "control_coeff_{}".format(i+1)) for i in range(len(self.control_components))], self.control_components)])
            treatment_output = sum([a * f(x) for a, f in zip([getattr(self, "treatment_coeff_{}".format(i+1)) for i in range(len(self.treatment_components))], self.treatment_components)])
            shared_output = sum([a * f(x) for a, f in zip([getattr(self, "shared_coeff_{}".format(i+1)) for i in range(len(self.shared_components))], self.shared_components)])
        else:
            control_output = sum([a * f(x) for a, f in zip(self.control_coeffs, self.control_components)])
            treatment_output = sum([a * f(x) for a, f in zip(self.treatment_coeffs, self.treatment_components)])
            shared_output = sum([a * f(x) for a, f in zip(self.shared_coeffs, self.shared_components)])
        output = torch.stack([control_output + shared_output, treatment_output + shared_output], dim=1)
        return torch.sigmoid(output) if self.output_binary else output

class AdditiveCMICK(nn.Module):
    """
    Class definition of the Additive Causal Multi-task Implicit Composite Kernel (CMICK)

    Arguments
    --------------
    components: List[CMICK], a list of CMICK objects to be added together
    component_assignment: List[List[int]], a list of lists of integers, where each list of integers specifies the 
        assignment of each input modality to each CMICK component. For example, say we have 3 components: f1, f2, and f3. 
        Let x[0] be the first input modality and x[1] be the second input modality. If component_assignment = 
        [[0],[1],[0,1]], this means the final output will be computed as:
        output = a1 * f1(x[0]) + a2 * f2(x[1]) + a3 * f3(x[0], x[1])
    coeffs: List[float], a list of coefficients for each CMICK component. If None, all coefficients will be set to 1.0
    coeff_trainable: bool, whether the coefficients are trainable parameters
    output_binary: bool, whether the output should be binary (i.e. sigmoid applied to the output)
    """
    def __init__(self, components: List[CMICK], component_assignment: List[List[int]], coeffs: List[float] = None, coeff_trainable: bool = False, 
                 output_binary: bool = False) -> None:
        super(AdditiveCMICK, self).__init__()
        self.components: nn.ModuleList = nn.ModuleList(components)
        self.component_assignment: List[List[int]] = component_assignment
        self.coeffs: List[float] = [1.0] * len(components) if coeffs is None else coeffs
        self.coeff_trainable: bool = coeff_trainable
        self.output_binary: bool = output_binary
        self._validate_inputs()
        if self.coeff_trainable:
            self._register_params()
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to Additive CMICK
        """
        assert all([isinstance(x, CMICK) for x in self.components]), "components must be a list of CMICK objects."
        assert all([isinstance(x, list) for x in self.component_assignment]), "component_assignment must be a list of lists."
        assert all([isinstance(x, float) for x in self.coeffs]), "coeffs must be a list of floats."
    
    def _register_params(self) -> None:
        """
        Register the coefficients as trainable parameters
        """
        for i in range(len(self.coeffs)):
            setattr(self, "coeff_{}".format(i+1), nn.Parameter(torch.tensor(self.coeffs[i], requires_grad=True)))
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of Additive CMICK
        """
        if self.coeff_trainable:
            output = sum([getattr(self, "coeff_{}".format(i+1)) * f([x[j] for j in self.component_assignment[i]]) for i, f in enumerate(self.components)])
        else:
            output = sum([a * f([x[j] for j in self.component_assignment[i]]) for a, i, f in zip(self.coeffs, range(len(self.components)), self.components)])
        return torch.sigmoid(output) if self.output_binary else output

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
