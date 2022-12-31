# metrics.py: a file containing the definition of metrics for model evaluation
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np

def policy_risk(y_test_pred: np.ndarray, y_test: np.ndarray, t_test: np.ndarray):
    """
    Calculate the policy risk of a causal model with binary treatment

    Arguments
    --------------
    y_test_pred: np.ndarray, a numpy array of shape (num_test_samples, 2) that contains the predicted control and 
        treatment outcomes
    y_test: np.ndarray, a numpy array that contains the factual outcomes
    t_test: np.ndarray, a numpy array that contains the group assignment of each data point
    """
    assert len(y_test_pred.shape) == 2 and y_test_pred.shape[1] == 2, "y_test_pred must be a 2D array with 2 columns"
    y_test, t_test = y_test.reshape(-1), t_test.reshape(-1)
    PiT0 = (1-np.argmax(y_test_pred, axis=1)) * (1-t_test)
    PiT1 = np.argmax(y_test_pred, axis=1) * t_test
    return 1.-np.sum(y_test*(PiT0+PiT1))/(np.sum(PiT0)+np.sum(PiT1))

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