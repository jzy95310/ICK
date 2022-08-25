# helpers.py: a file containing helper functions for data processing and training
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
from typing import Dict, Tuple, List
from .constants import *
from .data_generator import create_ick_data_generator

def train_val_test_split(x: List[np.ndarray], y: np.ndarray, train_range: Tuple = (0.0,0.5), 
                         val_range: Tuple = (0.5,0.6), test_range: Tuple = (0.6,1.0), 
                         shuffle_data: bool = False, random_seed: int = 2020) -> Tuple:
    """
    Split the data into train, validation, and test sets
    
    Arguments
    --------------
    x: List[np.ndarray], a list of tensors containing information from multiple sources
    y: np.ndarray, a tensor containing the targets/labels for prediction
    train_range: Tuple, the range of the data to be used for training
    val_range: Tuple, the range of the data to be used for validation
    test_range: Tuple, the range of the data to be used for testing
    shuffle_data: bool, whether to shuffle the data in advance
    random_seed: int, the random seed for shuffling the data
    """
    assert len(train_range) == 2 and 0 <= train_range[0] <= train_range[1] <= 1, \
        "The train_range must be a tuple of length 2 and 0 <= train_range[0] <= train_range[1] <= 1"
    assert len(val_range) == 2 and 0 <= val_range[0] <= val_range[1] <= 1, \
        "The val_range must be a tuple of length 2 and 0 <= val_range[0] <= val_range[1] <= 1"
    assert len(test_range) == 2 and 0 <= test_range[0] <= test_range[1] <= 1, \
        "The test_range must be a tuple of length 2 and 0 <= test_range[0] <= test_range[1] <= 1"
    
    if shuffle_data:
        np.random.seed(random_seed)
        np.random.shuffle(y)
        for item in x:
            np.random.seed(random_seed)
            np.random.shuffle(item)
    data_size = len(y)
    x_train, y_train = list(map(lambda item: item[int(train_range[0]*data_size):int(train_range[1]*data_size)], x)), \
        y[int(train_range[0]*data_size):int(train_range[1]*data_size)]
    x_val, y_val = list(map(lambda item: item[int(val_range[0]*data_size):int(val_range[1]*data_size)], x)), \
        y[int(val_range[0]*data_size):int(val_range[1]*data_size)]
    x_test, y_test = list(map(lambda item: item[int(test_range[0]*data_size):int(test_range[1]*data_size)], x)), \
        y[int(test_range[0]*data_size):int(test_range[1]*data_size)]
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_generators_from_data(x_train: List[np.ndarray], y_train: np.ndarray, x_val: List[np.ndarray], 
                                y_val: np.ndarray, x_test: List[np.ndarray], y_test: np.ndarray, 
                                train_batch_size: int = 50, val_batch_size: int = 300, 
                                test_batch_size: int = 300) -> Dict:
    """
    Create the data generators for the ICK model

    Arguments
    --------------
    train_batch_size: int, batch size of the data generator for training
    val_batch_size: int, batch size of the data generator for validation
    test_batch_size: int, batch size of the data generator for testing
    """
    train_data_generator = create_ick_data_generator(x_train, y_train, shuffle_dataloader=True, batch_size=train_batch_size)
    val_data_generator = create_ick_data_generator(x_val, y_val, shuffle_dataloader=False, batch_size=val_batch_size)
    test_data_generator = create_ick_data_generator(x_test, y_test, shuffle_dataloader=False, batch_size=test_batch_size)
    
    return {TRAIN: train_data_generator, VAL: val_data_generator, TEST: test_data_generator}

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