# data_generator.py: a file containing classes for generating data for ICK and other benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import numpy as np
from typing import List, Union, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    """
    Data generator class for ICK

    Arguments
    --------------
    x: Union[List[np.ndarray], np.ndarray], a list of tensors containing information from multiple sources or
        a single tensor containing information from a single source
    y: np.ndarray, a tensor containing the targets/labels for prediction
    dtype: str, the data type of the input tensors
    x_transform: a torchvision.transforms function that transforms all sources of information in x whose dimension
        are 3 or higher
    """
    def __init__(self, x: Union[List[np.ndarray], np.ndarray], y: np.ndarray, dtype: str = 'float32', 
                 x_transform: Union[Callable, None] = None) -> None:
        self.x: Union[List[np.ndarray], np.ndarray] = x
        self.y: np.ndarray = y
        self.dtype: str = dtype
        self.x_transform: Callable = x_transform
        self._validate_and_preprocess_inputs()
    
    def _validate_and_preprocess_inputs(self) -> None:
        """
        Validate and preprocess the inputs to the data generator
        """
        if not isinstance(self.x,List) and not isinstance(self.x,np.ndarray):
            raise TypeError("x must be a list of numpy arrays or a single numpy array")
        if len(set(map(len,self.x+[self.y] if isinstance(self.x,List) else [self.x, self.y]))) > 1:
            raise ValueError("All sources of information (including the target) must have the same number of samples.")
        # Make sure all sources of information in x are at least 2D and in correct data type
        if isinstance(self.x, list):
            self.x = tuple(map(lambda item: item.reshape(-1,1).astype(self.dtype) if len(item.shape) == 1 else item.astype(self.dtype), self.x))
        else:
            self.x = self.x.reshape(-1,1).astype(self.dtype) if len(self.x.shape) == 1 else self.x.astype(self.dtype)
        self.y = self.y.squeeze().astype(self.dtype)
        assert self.x_transform is None or isinstance(self.x_transform, Callable), "x_transform must be either None or a callable function."

    def __len__(self) -> int:
        return len(self.x[0]) if isinstance(self.x, tuple) else len(self.x)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(self.x, tuple):
            if self.x_transform is not None:
                return [self.x_transform(item[idx].astype(np.uint8)) if len(item[idx].shape) >= 3 else item[idx] for item in self.x], self.y[idx]
            else:
                return [item[idx] for item in self.x], self.y[idx]
        else:
            if self.x_transform is not None:
                return [self.x_transform(self.x[idx].astype(np.uint8)) if len(self.x[idx].shape) >= 3 else self.x[idx]], self.y[idx]
            else:
                return [self.x[idx]], self.y[idx]

def create_ick_data_generator(x: Union[List[np.ndarray], np.ndarray], y: np.ndarray, shuffle_dataloader: bool, batch_size: int, 
                              x_transform: Union[Callable, None] = None, drop_last: bool = False) -> DataLoader:
    """
    Function to create a data generator for ICK

    Arguments:
    --------------
    x: Union[List[np.ndarray], np.ndarray], a list of tensors containing information from multiple sources or
        a single tensor containing information from a single source
    y: np.ndarray, a tensor containing the targets/labels for prediction
    shuffle_dataloader: bool, whether to shuffle the data loader
    batch_size: int, batch size of the data generator
    x_transform: a torchvision.transforms function that transforms all sources of information in x whose dimension
        are 3 or higher
    drop_last: bool, whether to drop the last batch if it is smaller than the batch size

    Return:
    --------------
    A torch.utils.data.DataLoader which can generate data for learning ICK
    """
    dataset = DataGenerator(x, y, x_transform=x_transform)
    return DataLoader(dataset, shuffle=shuffle_dataloader, batch_size=batch_size, drop_last=drop_last)

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