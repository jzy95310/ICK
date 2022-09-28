# data_generator.py: a file containing the definition of data generators for benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import Callable, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataGeneratorForJointNN(Dataset):
    """
    Data generator class for joint models

    Arguments
    --------------
    x: np.ndarray, data to be passed into the neural network backbone of the joint model
    aug_feature: np.ndarray, the augmented feature to be concatenated with the latent representation from the 
        neural network backbone
    y: np.ndarray, the target/label for prediction
    y_pred_aug_feature: np.ndarray, the prediction from an ML model (e.g. random forest, SVM, etc.) using 
        the augmented feature
    dtype: str, the data type of the input tensors
    x_transform: None or Callable, a function to transform the input x
    """
    def __init__(self, x: np.ndarray, aug_feature: np.ndarray, y: np.ndarray, y_pred_aug_feature: np.ndarray, 
                 dtype: str = 'float32', x_transform: Union[Callable, None] = None) -> None:
        self.x: np.ndarray = x
        self.aug_feature: np.ndarray = aug_feature
        self.y: np.ndarray = y
        self.y_pred_aug_feature: np.ndarray = y_pred_aug_feature
        self.dtype: str = dtype
        self.x_transform: Union[Callable, None] = x_transform
        self._validate_and_preprocess_inputs()
    
    def _validate_and_preprocess_inputs(self) -> None:
        """
        Preprocess the inputs to the data generator
        Make sure that x and aug_feature are at least 2D and y and y_pred_aug_feature are 1D
        """
        self.x = self.x.reshape(-1,1).astype(self.dtype) if len(self.x.shape) == 1 else self.x.astype(self.dtype)
        self.aug_feature = self.aug_feature.reshape(-1,1).astype(self.dtype) if len(self.aug_feature.shape) == 1 else self.aug_feature.astype(self.dtype)
        self.y = self.y.reshape(-1).astype(self.dtype)
        self.y_pred_aug_feature = self.y_pred_aug_feature.reshape(-1).astype(self.dtype)
        assert self.x_transform is None or isinstance(self.x_transform, Callable), "x_transform must be either None or a callable function."
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.x_transform(self.x[idx].astype(np.uint8)) if self.x_transform is not None else self.x[idx]), \
            self.aug_feature[idx], self.y[idx], self.y_pred_aug_feature[idx]

def create_joint_nn_data_generator(x: np.ndarray, aug_feature: np.ndarray, y: np.ndarray, y_pred_aug_feature: np.ndarray, 
                                   shuffle_dataloader: bool, batch_size: int, x_transform: Union[Callable, None] = None) -> DataLoader:
    """
    Create a data generator for joint models

    Arguments
    --------------
    x: np.ndarray, data to be passed into the neural network backbone of the joint model
    aug_feature: np.ndarray, the augmented feature to be concatenated with the latent representation from the 
        neural network backbone
    y: np.ndarray, the target/label for prediction
    y_pred_aug_feature: np.ndarray, the prediction from an ML model (e.g. random forest, SVM, etc.) using 
        the augmented feature
    shuffle_dataloader: bool, whether to shuffle the data in the dataloader
    batch_size: int, the batch size for the data generator
    x_transform: None or Callable, a function to transform the input x
    """
    data_generator = DataGeneratorForJointNN(x, aug_feature, y, y_pred_aug_feature, x_transform=x_transform)
    return DataLoader(data_generator, batch_size=batch_size, shuffle=shuffle_dataloader)

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