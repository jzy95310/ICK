# helpers.py: a file containing helper functions for data processing, model creation, and training of ICK models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import os, sys
sys.path.insert(0, '..')
from pathlib import Path
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tarfile
import urllib.request
from scipy import stats
from sklearn import metrics
from torch import nn
from typing import Callable, Dict, Tuple, List, Union, Optional
from kernels.constants import *
from .constants import *
from .data_generator import create_ick_data_generator
from kernels.nn import ImplicitConvNet2DKernel, ImplicitNNKernel
from google_drive_downloader import GoogleDriveDownloader as gdd

def train_val_test_split(x: Union[List[np.ndarray], np.ndarray], y: np.ndarray, train_range: Tuple = (0.0,0.5), 
                         val_range: Tuple = (0.5,0.6), test_range: Tuple = (0.6,1.0), 
                         shuffle_data: bool = False, random_seed: int = 2020) -> Tuple:
    """
    Split the data into train, validation, and test sets
    
    Arguments
    --------------
    x: Union[List[np.ndarray], np.ndarray], a list of arrays containing information from multiple sources or
        a single array containing information from a single source
    y: np.ndarray, an array containing the targets/labels for prediction
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

    if isinstance(x, list):
        x_train = list(map(lambda item: item[int(train_range[0]*data_size):int(train_range[1]*data_size)], x))
        x_val = list(map(lambda item: item[int(val_range[0]*data_size):int(val_range[1]*data_size)], x))
        x_test = list(map(lambda item: item[int(test_range[0]*data_size):int(test_range[1]*data_size)], x))
    else:
        x_train = x[int(train_range[0]*data_size):int(train_range[1]*data_size)]
        x_val = x[int(val_range[0]*data_size):int(val_range[1]*data_size)]
        x_test = x[int(test_range[0]*data_size):int(test_range[1]*data_size)]
    y_train = y[int(train_range[0]*data_size):int(train_range[1]*data_size)]
    y_val = y[int(val_range[0]*data_size):int(val_range[1]*data_size)]
    y_test = y[int(test_range[0]*data_size):int(test_range[1]*data_size)]

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_generators_from_data(x_train: Union[List[np.ndarray], np.ndarray], y_train: np.ndarray, 
                                x_test: Union[List[np.ndarray], np.ndarray], y_test: np.ndarray, 
                                x_val: Union[List[np.ndarray], np.ndarray, None] = None, y_val: Union[np.ndarray, None] = None, 
                                train_batch_size: int = 32, val_batch_size: int = 64, test_batch_size: int = 64, 
                                x_transform: Union[Callable, None] = None, drop_last: bool = False) -> Dict:
    """
    Create the data generators for the ICK model

    Arguments
    --------------
    train_batch_size: int, batch size of the data generator for training
    val_batch_size: int, batch size of the data generator for validation
    test_batch_size: int, batch size of the data generator for testing
    x_transform: Union[Callable, None], a torchvision.transforms function that transforms all sources of information 
        in x whose dimension are 3 or higher
    drop_last: bool, whether to drop the last batch if it is smaller than the batch size
    """
    train_data_generator = create_ick_data_generator(x_train, y_train, shuffle_dataloader=True, batch_size=train_batch_size, 
                                                     x_transform=x_transform, drop_last=drop_last)
    test_data_generator = create_ick_data_generator(x_test, y_test, shuffle_dataloader=False, batch_size=test_batch_size, 
                                                    x_transform=x_transform, drop_last=drop_last)
    if x_val is not None and y_val is not None:
        val_data_generator = create_ick_data_generator(x_val, y_val, shuffle_dataloader=False, batch_size=val_batch_size, 
                                                       x_transform=x_transform, drop_last=drop_last)
    else:
        val_data_generator = None
    
    return {TRAIN: train_data_generator, VAL: val_data_generator, TEST: test_data_generator}

def calculate_stats(pred_vals_mean: np.ndarray, true_vals: np.ndarray, pred_vals_std: Union[np.ndarray, None] = None,
                    data_save_path: str = None) -> Tuple:
    """
    Calculate the R-squared values, RMSE, MAE, mean standardized log loss (MSLL), negative log predictive density (NLPD) 
    for evaluating the predictions
    
    Arguments
    --------------
    pred_vals_mean: np.ndarray, the mean of the predicted values
    true_vals: np.ndarray, the true values
    pred_vals_std: Union[np.ndarray, None], the standard deviation of the predicted values
    data_save_path: str, the path to save the passed-in data
    """
    def NLPD(y_pred_mean: np.ndarray, y_pred_std: np.ndarray, y_true: np.ndarray):
        logits = 1./np.sqrt(2*np.pi*y_pred_std**2) * np.exp(-(y_true-y_pred_mean)**2/(2*y_pred_std**2))
        for i in range(len(logits)):
            logits[i] = max(logits[i], np.finfo(np.float32).eps)
        return -np.mean(np.log(logits))
    
    if data_save_path is not None:
        if not os.path.exists(os.path.dirname(data_save_path)):
            os.makedirs(os.path.dirname(data_save_path))
        data_dict = {'true_vals': true_vals, 'pred_vals': pred_vals_mean}
        if pred_vals_std is not None:
            data_dict['pred_vals_std'] = pred_vals_std
        with open(data_save_path, 'wb') as fp:
            pkl.dump(data_dict, fp)
    spearmanr = stats.spearmanr(pred_vals_mean, true_vals)[0]
    pearsonr = stats.pearsonr(pred_vals_mean, true_vals)[0]
    rmse = np.sqrt(metrics.mean_squared_error(true_vals, pred_vals_mean))
    mae = metrics.mean_absolute_error(true_vals, pred_vals_mean)
    res = tuple([spearmanr, pearsonr, rmse, mae])
    if pred_vals_std is not None:
        msll_score = np.mean((true_vals-pred_vals_mean)**2/(2*pred_vals_std**2) + 0.5*np.log(2*np.pi*pred_vals_std**2))
        nlpd_score = NLPD(pred_vals_mean, pred_vals_std, true_vals)
        res = res + (msll_score, nlpd_score,)
    return res

def plot_pred_vs_true_vals(pred_vals: np.ndarray, true_vals: np.ndarray, x_label: str, y_label: str, title: str = None, 
                           fig_save_path: str = None, lower_bound: float = 0.0, upper_bound: float = 600.0, margin: float = 20.0, 
                           **kwargs) -> None:
    """
    Plot the predicted values against the true values
    
    Arguments
    --------------
    pred_vals: np.ndarray, the predicted values
    true_vals: np.ndarray, the true values
    x_label: str, the label for the x-axis
    y_label: str, the label for the y-axis
    title: str, the title of the plot
    fig_save_path: str, the path to save the plot
    lower_bound: float, the lower-left coordinates (both x and y) of the plot
    upper_bound: float, the upper-right coordinates (both x and y) of the plot
    margin: float, additional margin beyond the lower and upper bound for better visualization
    kwargs: additional texts to be displayed at the upper-left corner of the plot. For example, kwargs = {'RMSE': 0.5}
        will be displayed as "RMSE = 0.5"
    """
    plt.clf()
    _, ax = plt.subplots(figsize=(12, 10))
    data = pd.DataFrame(data={'true_vals': true_vals, 'pred_vals': pred_vals})
    ax = sns.histplot(data, x='true_vals', y='pred_vals', cbar=False, color='orange')
    ax.set_xlim((lower_bound-margin, upper_bound+margin))
    ax.set_ylim((lower_bound-margin, upper_bound+margin))
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'r--', lw=3)
    ax.set_xlabel(x_label, size = 30)
    ax.set_ylabel(y_label, size = 30)
    if title is not None:
        ax.set_title(title, size = 30)
    ax.tick_params(labelsize = 28)
    for i, (key, val) in enumerate(kwargs.items()):
        ax.text(0.02, 0.98-i*0.08, f'{key} = {val:.2f}', ha='left', va='top', color='black', weight='roman', 
                fontsize=30, transform=ax.transAxes)     
    if fig_save_path is not None:
        if not os.path.exists(os.path.dirname(fig_save_path)):
            os.makedirs(os.path.dirname(fig_save_path))
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.show()

def attach_single_output_dense_layer(model: ImplicitNNKernel, activation: str = 'relu', dropout_ratio: float = 0.0) -> ImplicitNNKernel:
    """
    Attach a single output dense layer to the model
    
    Arguments
    --------------
    model: the ICK model to attach the dense layer
    activation: str, the activation function for the previous dense layer
    dropout_ratio: float, the dropout ratio for the previous dense layer
    """
    assert isinstance(model, ImplicitNNKernel), 'The model must be an instance of ImplicitNNKernel.'
    if isinstance(model, ImplicitConvNet2DKernel):
        model.conv_blocks.append(ACTIVATIONS[activation])
        model.conv_blocks.append(nn.Dropout(dropout_ratio))
        model.conv_blocks.append(nn.Linear(model.conv_blocks[-3].out_features, 1))
    else:
        model.dense_blocks.append(ACTIVATIONS[activation])
        model.dense_blocks.append(nn.Dropout(dropout_ratio))
        model.dense_blocks.append(nn.Linear(model.dense_blocks[-3].out_features, 1))
    return model

def download_gdrive_if_needed(path: Path, file_id: str) -> None:
    """
    Helper for downloading a file from Google Drive, if it is now already on the disk.

    Parameters
    ----------
    path: Path
        Where to download the file
    file_id: str
        Google Drive File ID. Details: https://developers.google.com/drive/api/v3/about-files
    """
    path = Path(path)

    if path.exists():
        return

    gdd.download_file_from_google_drive(file_id=file_id, dest_path=path)


def download_http_if_needed(path: Path, url: str) -> None:
    """
    Helper for downloading a file, if it is now already on the disk.

    Parameters
    ----------
    path: Path
        Where to download the file.
    url: URL string
        HTTP URL for the dataset.
    """
    path = Path(path)

    if path.exists():
        return

    if url.lower().startswith("http"):
        urllib.request.urlretrieve(url, path)  # nosec
        return

    raise ValueError(f"Invalid url provided {url}")


def unarchive_if_needed(path: Path, output_folder: Path) -> None:
    """
    Helper for uncompressing archives. Supports .tar.gz and .tar.

    Parameters
    ----------
    path: Path
        Source archive.
    output_folder: Path
        Where to unarchive.
    """
    if str(path).endswith(".tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(path=output_folder)
        tar.close()
    elif str(path).endswith(".tar"):
        tar = tarfile.open(path, "r:")
        tar.extractall(path=output_folder)
        tar.close()
    else:
        raise NotImplementedError(f"archive not supported {path}")


def download_if_needed(
    download_path: Path,
    file_id: Optional[str] = None,  # used for downloading from Google Drive
    http_url: Optional[str] = None,  # used for downloading from a HTTP URL
    unarchive: bool = False,  # unzip a downloaded archive
    unarchive_folder: Optional[Path] = None,  # unzip folder
) -> None:
    """
    Helper for retrieving online datasets.

    Parameters
    ----------
    download_path: str
        Where to download the archive
    file_id: str, optional
        Set this if you want to download from a public Google drive share
    http_url: str, optional
        Set this if you want to download from a HTTP URL
    unarchive: bool
        Set this if you want to try to unarchive the downloaded file
    unarchive_folder: str
        Mandatory if you set unarchive to True.
    """
    if file_id is not None:
        download_gdrive_if_needed(download_path, file_id)
    elif http_url is not None:
        download_http_if_needed(download_path, http_url)
    else:
        raise ValueError("Please provide a download URL")

    if unarchive and unarchive_folder is None:
        raise ValueError("Please provide a folder for the archive")
    if unarchive and unarchive_folder is not None:
        unarchive_if_needed(download_path, unarchive_folder)

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