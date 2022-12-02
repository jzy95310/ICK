# metrics.py: a file containing the definition of metrics for model evaluation
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np

def policy_risk(y_test_pred: np.ndarray, y_test: np.ndarray, t_test: np.ndarray):
    """
    Calculate the policy risk of a causal model.

    Arguments
    --------------
    y_test_pred: np.ndarray, a numpy array of shape (num_test_samples, 2) that contains the predicted control and 
        treatment outcomes
    y_test: np.ndarray, a numpy array that contains the factual outcomes
    t_test: np.ndarray, a numpy array that contains the group assignment of each data point
    """
    assert len(y_test_pred.shape) == 2 and y_test_pred.shape[1] == 2, "y_test_pred must be a 2D array with 2 columns"
    y_test, t_test = y_test.reshape(-1), t_test.reshape(-1)
    N_test = len(y_test)
    N_Pi0, N_Pi1 = np.sum(1-np.argmax(y_test_pred, axis=1)), np.sum(np.argmax(y_test_pred, axis=1))
    PiT0 = (1-np.argmax(y_test_pred, axis=1)) * (1-t_test)
    PiT1 = np.argmax(y_test_pred, axis=1) * t_test
    # N_PiT0, N_PiT1 = np.sum(PiT0), np.sum(PiT1)
    # print("N_Pi0: {}, N_Pi1: {}, N_PiT0: {}, N_PiT1: {}".format(N_Pi0, N_Pi1, N_PiT0, N_PiT1))
    # return np.mean(1.-(1./N_PiT0*(PiT0*y_test*N_Pi0/N_test) + 1./N_PiT1*(PiT1*y_test*N_Pi1/N_test)))
    # return 1.-(1./N_PiT0*np.mean(PiT0*y_test*N_Pi0/N_test) + 1./N_PiT1*np.mean(PiT1*y_test*N_Pi1/N_test))
    return 1.-(np.mean(PiT0*y_test)*N_Pi0/N_test + np.mean(PiT1*y_test)*N_Pi1/N_test)

def att_err(y_test_pred: np.ndarray, y_test: np.ndarray, t_test: np.ndarray):
    """
    Calculate the average treatment effect error of a causal model.

    Arguments
    --------------
    y_test_pred: np.ndarray, a numpy array of shape (num_test_samples, 2) that contains the predicted control and 
        treatment outcomes
    y_test: np.ndarray, a numpy array that contains the factual outcomes
    t_test: np.ndarray, a numpy array that contains the group assignment of each data point
    """
    assert len(y_test_pred.shape) == 2 and y_test_pred.shape[1] == 2, "y_test_pred must be a 2D array with 2 columns"
    y_test, t_test = y_test.reshape(-1), t_test.reshape(-1)
    N_T0, N_T1 = np.sum(1-t_test), np.sum(t_test)
    att = np.sum(t_test*y_test)/N_T1 - np.sum((1-t_test)*y_test)/N_T0
    return abs(att-np.sum((y_test_pred[:,1]-y_test_pred[:,0])*t_test)/N_T1)

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