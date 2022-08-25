# data_generator.py: a file containing classes for generating data for ICK and other benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    """
    Data generator class for ICK

    Arguments
    --------------
    x: List[np.ndarray], an input tensor containing information from multiple sources
    y: np.ndarray, a tensor containing the targets/labels for prediction
    dtype: str, the data type of the input tensors
    shuffle: bool, whether to shuffle the data
    random_seed: int, the random seed for shuffling the data
    """
    def __init__(self, x: List[np.ndarray], y: np.ndarray, dtype: str = 'float32') -> None:
        self.x: List[np.ndarray] = x
        self.y: np.ndarray = y
        self.dtype: str = dtype
        self._validate_and_preprocess_inputs()
    
    def _validate_and_preprocess_inputs(self) -> None:
        """
        Validate and preprocess the inputs to the data generator
        """
        if not isinstance(self.x,List):
            raise TypeError("The input array must be a List even if it only contains one source of information.")
        if len(set(map(len,self.x+[self.y]))) > 1:
            raise ValueError("All sources of information (including the target) must have the same number of samples.")
        # Make sure all sources of information in x are at least 2D and in correct data type
        self.x = tuple(map(lambda item: item.reshape(-1,1).astype(self.dtype) if len(item.shape) == 1 else item.astype(self.dtype), self.x))

    def __len__(self) -> int:
        return len(self.x[0])

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [item[idx] for item in self.x], self.y[idx]

def create_ick_data_generator(x: List[np.ndarray], y: np.ndarray, shuffle_dataloader: bool, batch_size: int):
    """
    Function to create a data generator for ICK

    Arguments:
    --------------
    x: List[np.ndarray], a list of tensors containing information from multiple sources
    y: np.ndarray, a tensor containing the targets/labels for prediction
    shuffle_dataloader: bool, whether to shuffle the data loader
    batch_size: int, batch size of the data generator

    Return:
    --------------
    A torch.utils.data.DataLoader which can generate data for learning ICK
    """
    dataset = DataGenerator(x,y)
    return DataLoader(dataset, shuffle=shuffle_dataloader, batch_size=batch_size)

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