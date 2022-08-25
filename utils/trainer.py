# trainer.py: a file containing classes for training the ICK model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch.utils.data import DataLoader

import os, sys
import logging
import time
from typing import Dict, List, Tuple
from .constants import *

class Trainer:
    """
    Class for training the ICK model

    Arguments
    --------------
    model: torch.nn.Module, the ICK model to be trained
    data_generators: Dict, a dict of data generators for the ICK model where keys must be 'train', 'val', and 'test'
    optimizer: torch.optim.Optimizer, the optimizer for learning the ICK model
    model_save_dir: str, the directory to save the trained ICK model
    model_name: str, the name of the trained ICK model
    loss_fn: torch.nn.modules.loss._Loss, the loss function for optimizing the ICK model
    device: torch.device, the device to train the model on
    epochs: int, the number of epochs to train the model for
    patience: int, the number of epochs to wait before early stopping
    logger: logging.Logger, an instance of logging.Logger for logging messages, errors, exceptions
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optimizer: torch.optim.Optimizer,
                 model_save_dir: str, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 device: torch.device = torch.device('cpu'), epochs: int = 100, patience: int = 10, 
                 logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.model: torch.nn.Module = model
        self.data_generators: Dict = data_generators
        self.optimizer: torch.optim.Optimizer = optimizer
        self.model_save_dir: str = model_save_dir
        self.model_name: str = model_name
        self.loss_fn: torch.nn.modules.loss._Loss = loss_fn
        self.device: torch.device = device
        self.epochs: int = epochs
        self.patience: int = patience
        self.logger: logging.Logger = logger
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler(stream=sys.stderr))
        self._validate_inputs()
    
    def _assign_device_to_data(self, data: List, target: torch.Tensor) -> Tuple:
        """
        Assign the device to the data and target
        """
        data = list(map(lambda x: x.to(self.device), data))
        target = target.to(self.device)
        return data, target
    
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the trainer
        """
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if not isinstance(self.data_generators, Dict):
            raise TypeError("data_generators must be a dictionary")
        if set(self.data_generators.keys()) != {TRAIN, VAL, TEST}:
            raise ValueError("data_generators must contain keys \'train\', \'val\', and \'test\'")
        if not all([type(x) is torch.utils.data.DataLoader for x in self.data_generators.values()]):
            raise TypeError("data_generators\' values must be instances of torch.utils.data.DataLoader")
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be an instance of torch.optim.Optimizer")
        if not isinstance(self.loss_fn, torch.nn.modules.loss._Loss):
            raise TypeError("loss_fn must be an instance of torch.nn.modules.loss._Loss")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be an instance of torch.device")
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be an instance of logging.Logger")
    
    def train(self) -> None:
        """
        Train the ICK model
        """
        # Early stopping
        best_loss = 1e9
        best_model_state_dict = None
        trigger_times = 0

        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            # Training
            self.model.to(self.device)
            y_train_pred = torch.empty(0).to(self.device)
            y_train_true = torch.empty(0).to(self.device)
            self.model.train()
            train_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7.6f}")
            for step, batch in enumerate(self.data_generators[TRAIN]):
                data, target = self._assign_device_to_data(batch[0], batch[1])
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward and backward pass
                output = self.model(data).float()
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                # Record the predictions
                y_train_pred = torch.cat((y_train_pred, output), dim=0)
                y_train_true = torch.cat((y_train_true, target), dim=0)
            # Log the training time and loss
            train_time = time.time() - train_start
            train_loss = self.loss_fn(y_train_pred, y_train_true).item()
            self.logger.info("{:.0f}s for {} steps - {:.0f}ms/step - loss {:.4f}" \
                  .format(train_time, step + 1, train_time * 1000 // (step + 1), train_loss))
            # Validation
            val_start = time.time()
            self.logger.info("Validation:")
            val_loss = self.validate()
            val_time = time.time() - val_start
            self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
            # Early stopping
            if val_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best model
                    self.logger.info("Early stopping - patience reached")
                    self.logger.info("Saving the best model")
                    torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                best_loss = val_loss
                best_model_state_dict = self.model.state_dict()
        if trigger_times < self.patience:
            self.logger.info("Training completed without early stopping.")
    
    def validate(self) -> float:
        """
        Evaluate the ICK model on the validation data
        """
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[VAL]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                output = self.model(data).float()
                y_val_pred = torch.cat((y_val_pred, output), dim=0)
                y_val_true = torch.cat((y_val_true, target), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true).item()
        return val_loss

    def predict(self) -> Tuple:
        """
        Evaluate the ICK model on the test data
        """
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                output = self.model(data).float()
                y_test_pred = torch.cat((y_test_pred, output), dim=0)
                y_test_true = torch.cat((y_test_true, target), dim=0)
        return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

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