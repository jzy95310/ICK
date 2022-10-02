# train_benchmarks.py: a file containing helper functions for data processing, model creation, and training of benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import os, sys
import numpy as np
import logging
from torch import nn
import vit_pytorch as vitorch
from typing import Tuple, Union, Dict, Callable
from utils.helpers import train_val_test_split
from utils.constants import *
from .data_generator import create_joint_nn_data_generator

def train_val_test_split_for_joint_nn(x: np.ndarray, aug_feature: np.ndarray, y: np.ndarray, y_pred_aug_feature: np.ndarray, 
                                      train_range: Tuple = (0.0,0.5), val_range: Tuple = (0.5,0.6), test_range: Tuple = (0.6,1.0), 
                                      shuffle_data: bool = False, random_seed: int = 2020) -> Tuple:
    """
    Split the data into train, validation, and test sets
    
    Arguments
    --------------
    x: np.ndarray, data to be passed into the neural network backbone of the joint model
    aug_feature: np.ndarray, the augmented feature to be concatenated with the latent representation from the 
        neural network backbone
    y: np.ndarray, the target label for prediction
    y_pred_aug_feature: np.ndarray, the prediction from an ML model (e.g. random forest, SVM, etc.) using 
        the augmented feature
    train_range: Tuple, the range of the data to be used for training
    val_range: Tuple, the range of the data to be used for validation
    test_range: Tuple, the range of the data to be used for testing
    shuffle_data: bool, whether to shuffle the data in advance
    random_seed: int, the random seed for shuffling the data
    """
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y, train_range, val_range, test_range, shuffle_data, random_seed)
    if shuffle_data:
        np.random.seed(random_seed)
        np.random.shuffle(aug_feature)
        np.random.seed(random_seed)
        np.random.shuffle(y_pred_aug_feature)
    data_size = len(y)

    aug_feature_train = aug_feature[int(train_range[0]*data_size):int(train_range[1]*data_size)]
    aug_feature_val = aug_feature[int(val_range[0]*data_size):int(val_range[1]*data_size)]
    aug_feature_test = aug_feature[int(test_range[0]*data_size):int(test_range[1]*data_size)]
    y_pred_train = y_pred_aug_feature[int(train_range[0]*data_size):int(train_range[1]*data_size)]
    y_pred_val = y_pred_aug_feature[int(val_range[0]*data_size):int(val_range[1]*data_size)]
    y_pred_test = y_pred_aug_feature[int(test_range[0]*data_size):int(test_range[1]*data_size)]

    return x_train, aug_feature_train, y_train, y_pred_train, \
        x_val, aug_feature_val, y_val, y_pred_val, \
        x_test, aug_feature_test, y_test, y_pred_test

def create_generators_from_data_for_joint_nn(x_train: np.ndarray, aug_feature_train: np.ndarray, y_train: np.ndarray, y_pred_train: np.ndarray, 
                                             x_test: np.ndarray, aug_feature_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray, 
                                             x_val: Union[np.ndarray, None] = None, aug_feature_val: Union[np.ndarray, None] = None, 
                                             y_val: Union[np.ndarray, None] = None, y_pred_val: Union[np.ndarray, None] = None, 
                                             train_batch_size: int = 32, val_batch_size: int = 64, test_batch_size: int = 64, 
                                             x_transform: Union[Callable, None] = None) -> Dict:
    """
    Create the data generators for benchmark joint NN model

    Arguments
    --------------
    train_batch_size: int, batch size of the data generator for training
    val_batch_size: int, batch size of the data generator for validation
    test_batch_size: int, batch size of the data generator for testing
    x_transform: None or Callable, a function to transform the input x
    """
    train_data_generator = create_joint_nn_data_generator(x_train, aug_feature_train, y_train, y_pred_train, shuffle_dataloader=True, 
                                                          batch_size=train_batch_size, x_transform=x_transform)
    test_data_generator = create_joint_nn_data_generator(x_test, aug_feature_test, y_test, y_pred_test, shuffle_dataloader=False, 
                                                         batch_size=test_batch_size, x_transform=x_transform)
    if all([m is not None for m in [x_val, aug_feature_val, y_val, y_pred_val]]):
        val_data_generator = create_joint_nn_data_generator(x_val, aug_feature_val, y_val, y_pred_val, shuffle_dataloader=False, 
                                                            batch_size=val_batch_size, x_transform=x_transform)
    else:
        val_data_generator = None
    
    return {TRAIN: train_data_generator, VAL: val_data_generator, TEST: test_data_generator}

def pretrain_encoder_with_mae(encoder: nn.Module, dataloader: torch.utils.data.DataLoader, masking_ratio: float = 0.75, 
                              epochs: int = 500, decoder_dim: int = 512, decoder_depth: int = 6, patience: int = 5, 
                              data_idx: int = 0, model_name: str = 'encoder.pt', device: torch.device = torch.device('cpu'), 
                              logger: logging.Logger = logging.getLogger("Trainer")) -> None:
    """
    Pretrain the passed-in encoder with Masked Autoencoder (MAE) and save the pre-trained encoder

    Arguments
    --------------
    encoder: nn.Module, the encoder to be pretrained
    dataloader: torch.utils.data.DataLoader, the dataloader for the pre-training
    masking_ratio: float, the masking ratio of MAE, default to 0.75
    epochs: int, the number of epochs for pre-training, default to 500
    decoder_dim: int, the dimension of the decoder, default to 500
    decoder_depth: int, the depth of the decoder, default to 6
    patience: int, the patience for early stopping, default to 5
    data_idx: int, the index of the data in batch to train the MAE, default to 0
    model_name: str, the name of the saved pre-trained model
    device: torch.device, the device to run the pre-training
    logger: logging.Logger, the logger for logging the pre-training process
    """
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    mae = vitorch.MAE(
        encoder=encoder,
        masking_ratio=masking_ratio, 
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth
    ).to(device)
    loss_smallest, early_stopping_count = 1e9, 0
    logger.info("Start pre-training:")
    for epoch in range(epochs):
        logger.info("Pre-training epoch {}".format(epoch))
        for batch in dataloader:
            data = batch[data_idx].to(device)
            loss = mae(data)
            loss.backward()
        logger.info("Pre-training loss: {}".format(loss))
        if loss < loss_smallest:
            early_stopping_count = 0
            loss_smallest = loss
            if not os.path.exists('./pretrained_encoder_checkpoints'):
                os.makedirs('./pretrained_encoder_checkpoints')
            torch.save(mae.encoder.state_dict(), './pretrained_encoder_checkpoints/pretrained_{}'.format(model_name))
        else:
            early_stopping_count += 1
            if early_stopping_count > patience:
                print('Early stopping criterion reached. Exiting...')
                break

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