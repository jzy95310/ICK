# trainer.py: a file containing classes for training the ICK model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
import torchbnn as bnn

import os, sys
sys.path.insert(0, '..')
import numpy as np
import logging
import time
import inspect
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score
from .constants import *
from .helpers import calculate_stats, plot_pred_vs_true_vals
from .losses import *
from kernels.nystrom import *
from joblib import Parallel, delayed

class BaseTrainer(ABC):
    """
    Base class for trainer

    Arguments
    --------------
    model: Union[torch.nn.Module, List], the ICK model or an ensemble of ICK models to be trained
    data_generators: Dict, a dict of data generators for the ICK model where keys must be 'train', 'val', and 'test'
    optim: str, the name of the optimizer to use for training the ICK model
    optim_params: Dict, a dict of parameters for the optimizer
    lr_scheduler: torch.optim.lr_scheduler, the learning rate scheduler to use for training the ICK model
    model_save_dir: str, the directory to save the trained ICK model. If None, the model will not be saved
    model_name: str, the name of the trained ICK model
    loss_fn: torch.nn.modules.loss._Loss, the loss function for optimizing the ICK model
    device: torch.device, the device to train the model on
    validation: bool, whether to validate the model during training, default to True
    epochs: int, the number of epochs to train the model for
    patience: int, the number of epochs to wait before early stopping
    verbose: int, the level of verbosity for the trainer, default to 0.
        verbose = 0: no logging
        verbose = 1: log all statistics of test predictions
        verbose = 2: log and plot all statistics of test predictions
    logger: logging.Logger, an instance of logging.Logger for logging messages, errors, exceptions
    """
    @abstractmethod
    def __init__(self, model: Union[torch.nn.Module, List], data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), device: torch.device = torch.device('cpu'), 
                 validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.model: Union[torch.nn.Module, List] = model
        self.data_generators: Dict = data_generators
        self.optim: str = optim
        self.optim_params: Dict = optim_params
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler
        self.model_save_dir: str = model_save_dir
        self.model_name: str = model_name
        self.loss_fn: torch.nn.modules.loss._Loss = loss_fn
        self.device: torch.device = device
        self.validation: bool = validation
        self.epochs: int = epochs
        self.patience: int = patience
        self.logger: logging.Logger = logger
        self.verbose: int = verbose
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        if self.model_save_dir is not None and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self._validate_inputs()
        self._set_optimizer()
    
    def _assign_device_to_data(self, data: Union[List, torch.Tensor], target: torch.Tensor) -> Tuple:
        """
        Assign the device to the data and target
        """
        if isinstance(data, list):
            data = list(map(lambda x: x.to(self.device), data))
        else:
            data = data.to(self.device)
        target = target.to(self.device)
        return data, target
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the trainer
        """
        if not isinstance(self.data_generators, Dict):
            raise TypeError("data_generators must be a dictionary.")
        if not set(self.data_generators.keys()).issubset({TRAIN, VAL, TEST}):
            raise ValueError("The keys of data_generators must be a subset of {\'train\', \'val\', and \'test\'}")
        if self.optim not in OPTIMIZERS:
            raise TypeError("The optimizer must be one of the following: {}".format(OPTIMIZERS.keys()))
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise TypeError("lr_scheduler must be an instance of torch.optim.lr_scheduler._LRScheduler")
        if not isinstance(self.loss_fn, torch.nn.modules.loss._Loss):
            raise TypeError("loss_fn must be an instance of torch.nn.modules.loss._Loss")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be an instance of torch.device")
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be an instance of logging.Logger")
    
    def _set_optimizer(self) -> None:
        """
        Set the optimizer for the trainer
        """
        self.optimizer = OPTIMIZERS[self.optim](self.model.parameters(), **self.optim_params)
    
    def _log_prediction_stats(self, pred_mean: np.ndarray, target: np.ndarray, pred_std: Union[np.ndarray, None] = None) -> Tuple:
        """
        Log and return the statistics of the predictions
        """
        if isinstance(self.loss_fn, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
            res_dict = {"AUC": roc_auc_score(target, pred_mean)}
        else:
            res = calculate_stats(pred_mean, target, pred_std)
            res_dict = {"Spearman_R": res[0], "Pearson_R": res[1], "RMSE": res[2], "MAE": res[3]}
            if len(res) > 4:
                res_dict["MSLL"] = res[4]
        for k, v in res_dict.items():
            self.logger.info("{}: {}".format(k, v))
        self.logger.info("\n")
        return res_dict
    
    def _plot_predictions(self, pred_mean: np.ndarray, target: np.ndarray, stats: Dict = None) -> None:
        """
        Plot the predictions
        """
        stats = stats if stats is not None else {}
        plot_pred_vs_true_vals(
            pred_mean, 
            target,
            x_label='Predicted Values',
            y_label='True Values',
            **stats
        )
    
    @abstractmethod
    def train(self) -> None:
        """
        Train the ICK model
        """
        pass

    @abstractmethod
    def predict(self) -> None:
        """
        Evaluate the ICK model on the test data
        """
        pass

class Trainer(BaseTrainer):
    """
    Class for training the ICK model

    Arguments
    --------------
    stop_criterion: str, the criterion to use for early stopping, default to 'loss'
        stop_criterion = 'loss': early stopping based on the validation loss
        stop_criterion = 'auc': early stopping based on the validation AUC, only applicable for binary classification
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 stop_criterion: str = 'loss', logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.stop_criterion: str = stop_criterion
        super(Trainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, loss_fn, 
                                      device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("The model must be an instance of torch.nn.Module")
        if self.stop_criterion not in ['loss', 'auc']:
            raise ValueError("stop_criterion must be one of the following: ['loss', 'auc']")
        super(Trainer, self)._validate_inputs()
    
    def _train_step(self) -> Tuple:
        """
        Perform a single training step
        """
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        self.model.to(self.device)
        self.model.train()
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
        train_loss = self.loss_fn(y_train_pred, y_train_true).item()
        return train_loss, step
    
    def train(self) -> None:
        # initialize the early stopping counter
        best_loss, best_auc = 1e9, 0
        best_model_state_dict = None
        trigger_times = 0

        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            # Training
            train_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7.6f}")
            # Log the training time and loss
            train_loss, step = self._train_step()
            train_time = time.time() - train_start
            self.logger.info("{:.0f}s for {} steps - {:.0f}ms/step - loss {:.4f}" \
                  .format(train_time, step + 1, train_time * 1000 // (step + 1), train_loss))
            # Validation
            val_start = time.time()
            self.logger.info("Validation:")
            val_loss = self.validate()
            val_time = time.time() - val_start
            self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Early stopping
            if self.stop_criterion == 'loss':
                early_stopping_flag = val_loss > best_loss
            else:
                val_auc = roc_auc_score(self.y_val_true.cpu().numpy(), self.y_val_pred.cpu().numpy())
                early_stopping_flag = val_auc < best_auc
            if early_stopping_flag:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best model
                    self.logger.info("Early stopping - patience reached")
                    if best_model_state_dict is not None:
                        self.logger.info("Restoring the best model")
                        self.model.load_state_dict(best_model_state_dict)
                    if self.model_save_dir is not None:
                        self.logger.info("Saving the best model")
                        torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                if self.stop_criterion == 'loss':
                    best_loss = val_loss
                else:
                    best_auc = val_auc
                best_model_state_dict = self.model.state_dict()
            # Visualize the test predictions if verbose > 0
            if self.verbose > 0:
                with torch.no_grad():
                    y_test_pred, y_test_true = self.predict()
                stats_for_test_pred = self._log_prediction_stats(y_test_pred, y_test_true)
                if self.verbose > 1:
                    self._plot_predictions(y_test_pred, y_test_true, stats_for_test_pred)
        if trigger_times < self.patience:
            self.logger.info("Training completed without early stopping.")
    
    def validate(self) -> float:
        """
        Evaluate the ICK model on the validation data
        """
        self.y_val_pred = torch.empty(0).to(self.device)
        self.y_val_true = torch.empty(0).to(self.device)
        self.model.eval()

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for batch in self.data_generators[key]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                output = self.model(data).float()
                self.y_val_pred = torch.cat((self.y_val_pred, output), dim=0)
                self.y_val_true = torch.cat((self.y_val_true, target), dim=0)
        val_loss = self.loss_fn(self.y_val_pred, self.y_val_true).item()
        return val_loss
    
    def _get_uncertainty(self) -> Tuple:
        """
        Return the uncertainty of the predictions based on the fitted kernel parameters
        ******************
        Work in progress
        ******************
        """
        y_test_pred_std = torch.empty(0).to(self.device)
        self.model.eval()
        fullargs, _, _, defaults, _, _, _ = inspect.getfullargspec(self.model.kernels[0].kernel_func)
        kwargs = fullargs[-len(defaults):]
        params = [x.item() for x in list(self.model.kernels[0].parameters())]
        params_dict = {k: v for k, v in zip(kwargs, params)}

        with torch.no_grad():
            x_train = torch.tensor(self.data_generators[TRAIN].dataset.x).to(self.device)
            K_xx = self.model.kernels[0].kernel_func(x_train, x_train, **params_dict)   # Use inducing points?
            K_xx_inv = torch.linalg.inv(K_xx)
            x_test = torch.tensor(self.data_generators[TEST].dataset.x).to(self.device)
            K_tx = self.model.kernels[0].kernel_func(x_test, x_train, **params_dict)
            K_tt = self.model.kernels[0].kernel_func(x_test, x_test, **params_dict)
            y_test_pred_std = torch.sqrt(torch.diag(K_tt - K_tx @ K_xx_inv @ K_tx.T))
        
        return y_test_pred_std

    def predict(self, uncertainty=False) -> Tuple:
        if uncertainty:
            assert len(self.model.kernels) == 1 and isinstance(self.model.kernels[0], ImplicitNystromKernel), \
                "The uncertainty feature is only available for the ICK model with a single implicit Nystrom kernel."
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                output = self.model(data).float()
                y_test_pred = torch.cat((y_test_pred, output), dim=0)
                y_test_true = torch.cat((y_test_true, target), dim=0)
        if not uncertainty:
            return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()
        else:
            y_test_pred_std = self._get_uncertainty()
            lower, upper = y_test_pred - y_test_pred_std, y_test_pred + y_test_pred_std
            return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy(), \
                lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

class VariationalBayesTrainer(BaseTrainer):
    """
    Class for training the Bayesian variant of ICK model

    Arguments
    --------------
    kl_weight: float, the weight of the KL divergence term in the loss function
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, logger: logging.Logger = logging.getLogger("Trainer"), 
                 patience: int = 10, verbose: int = 0, kl_weight: float = 0.1) -> None:
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight: float = kl_weight
        super(VariationalBayesTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                                      loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("The model must be an instance of torch.nn.Module")
        assert self.kl_weight >= 0, "kl_weight must be non-negative."
        super(VariationalBayesTrainer, self)._validate_inputs()
    
    def train(self) -> None:
        """
        No validation step and early stopping for training the Bayesian variant of ICK
        """
        # initialize the early stopping counter
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
                loss = self.loss_fn(output, target) + self.kl_loss(self.model) * self.kl_weight
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
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Early stopping
            if train_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best model
                    self.logger.info("Early stopping - patience reached")
                    if best_model_state_dict is not None:
                        self.logger.info("Restoring the best model")
                        self.model.load_state_dict(best_model_state_dict)
                    if self.model_save_dir is not None:
                        self.logger.info("Saving the best model")
                        torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                best_loss = train_loss
                best_model_state_dict = self.model.state_dict()
            # Visualize the test predictions if verbose > 0
            if self.verbose > 0:
                with torch.no_grad():
                    y_test_pred_mean, y_test_pred_std, y_test_true = self.predict()
                stats_for_test_pred = self._log_prediction_stats(y_test_pred_mean, y_test_true, y_test_pred_std)
                if self.verbose > 1:
                    self._plot_predictions(y_test_pred_mean, y_test_true, stats_for_test_pred)
        if trigger_times < self.patience:
            self.logger.info("Training completed.")
    
    def predict(self, num_samples: int = 500) -> None:
        """
        Evaluate the Bayesian variant of ICK model on the test data

        Arguments
        --------------
        num_samples: int, the number of samples to draw from the predictive posterior
        """
        y_test_pred = [torch.empty(0).to(self.device) for _ in range(num_samples)]
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        key = TRAIN if not self.validation else TEST
        with torch.no_grad():
            for i in range(num_samples):
                for batch in self.data_generators[key]:
                    data, target = self._assign_device_to_data(batch[0], batch[1])
                    output = self.model(data).float()
                    y_test_pred[i] = torch.cat((y_test_pred[i], output), dim=0)
                    if i == 0:
                        y_test_true = torch.cat((y_test_true, target), dim=0)
        y_test_pred_mean = torch.mean(torch.stack(y_test_pred, dim=0), dim=0)
        y_test_pred_std = torch.std(torch.stack(y_test_pred, dim=0), dim=0)
        return y_test_pred_mean.detach().cpu().numpy(), y_test_pred_std.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class EnsembleTrainer(BaseTrainer):
    """
    Class for training an ensemble of ICK models
    Note that each predictor in the ensemble is trained independently

    Arguments
    --------------
    num_jobs: int, the number of jobs to run in parallel
    stop_criterion: str, the criterion to use for early stopping, default to 'loss'
        stop_criterion = 'loss': early stopping based on the validation loss
        stop_criterion = 'auc': early stopping based on the validation AUC, only applicable for binary classification
    """
    def __init__(self, model: List, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 num_jobs: int = None, model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 stop_criterion: str = 'loss', logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.num_jobs: int = num_jobs
        self.stop_criterion: str = stop_criterion
        super(EnsembleTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                              loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        if not isinstance(self.model, List):
            raise TypeError("The ensemble model must be a List.")
        if self.stop_criterion not in ['loss', 'auc']:
            raise ValueError("stop_criterion must be one of the following: ['loss', 'auc']")
        super(EnsembleTrainer, self)._validate_inputs()

    def _set_optimizer(self) -> None:
        """
        Set the optimizer for the ensemble
        """
        self.optimizers = []
        for i in range(len(self.model)):
            self.optimizers.append(OPTIMIZERS[self.optim](self.model[i].parameters(), **self.optim_params))
    
    def _train_step(self, base_learner_idx: int) -> float:
        """
        Perform a single training step for a baselearner in the ensemble

        Arguments
        --------------
        base_learner_idx: int, the index of the baselearner in the ensemble
        """
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        self.model[base_learner_idx].train()
        self.model[base_learner_idx].to(self.device)
        for _, batch in enumerate(self.data_generators[TRAIN]):
            data, target = self._assign_device_to_data(batch[0], batch[1])
            # Zero the gradients
            self.optimizers[base_learner_idx].zero_grad()
            # Forward and backward pass
            output = self.model[base_learner_idx](data).float()
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizers[base_learner_idx].step()
            # Record the predictions
            y_train_pred = torch.cat((y_train_pred, output), dim=0)
            y_train_true = torch.cat((y_train_true, target), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true).item()
        return train_loss
    
    def train(self) -> None:
        """
        Train the ensemble of ICK models
        """
        # initialize the early stopping counter
        best_loss, best_auc = 1e9, 0
        best_model_state_dict = None
        trigger_times = 0

        self.logger.info("Training started:\n")
        # Parallelize the training
        with Parallel(n_jobs=self.num_jobs, prefer='threads', verbose=1) as parallel:
            for epoch in range(self.epochs):
                # Training
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
                self.logger.info(f"Learning rate: {self.optimizers[0].param_groups[0]['lr']:7.6f}")
                train_start = time.time()
                train_losses = parallel(delayed(self._train_step)(i) for i in range(len(self.model)))
                # Log the training time and loss
                train_time = time.time() - train_start
                train_loss = np.mean(train_losses)
                self.logger.info("Training time - {:.0f}s - loss {:.4f}".format(train_time, train_loss))
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                # Validation
                val_start = time.time()
                self.logger.info("Validation:")
                val_loss = self.validate()
                val_time = time.time() - val_start
                self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                # Early stopping
                if self.stop_criterion == 'loss':
                    early_stopping_flag = val_loss > best_loss
                else:
                    val_auc = roc_auc_score(self.y_val_true.cpu().numpy(), self.y_val_pred.cpu().numpy())
                    early_stopping_flag = val_auc < best_auc
                if early_stopping_flag:
                    trigger_times += 1
                    if trigger_times >= self.patience:
                        # Trigger early stopping and save the best ensemble
                        self.logger.info("Early stopping - patience reached")
                        if best_model_state_dict is not None:
                            self.logger.info("Restoring the best model")
                            for i in range(len(self.model)):
                                self.model[i].load_state_dict(best_model_state_dict['model_'+str(i)])
                        if self.model_save_dir is not None:
                            self.logger.info("Saving the best ensemble")
                            torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                        break
                else:
                    trigger_times = 0
                    if self.stop_criterion == 'loss':
                        best_loss = val_loss
                    else:
                        best_auc = val_auc
                    best_model_state_dict = {'model_'+str(i): self.model[i].state_dict() for i in range(len(self.model))}
                # Visualize the test predictions if verbose > 0
                if self.verbose > 0:
                    with torch.no_grad():
                        y_test_pred_mean, y_test_pred_std, y_test_true = self.predict()
                    stats_for_test_pred = self._log_prediction_stats(y_test_pred_mean, y_test_true, y_test_pred_std)
                    if self.verbose > 1:
                        self._plot_predictions(y_test_pred_mean, y_test_true, stats_for_test_pred)
            if trigger_times < self.patience:
                self.logger.info("Training completed.")
    
    def validate(self) -> float:
        """
        Evaluate the ICK ensemble model on the validation data
        """
        self.y_val_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        self.y_val_true = torch.empty(0).to(self.device)

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[key]:
                    data, target = self._assign_device_to_data(batch[0], batch[1])
                    output = self.model[i](data).float()
                    self.y_val_pred[i] = torch.cat((self.y_val_pred[i], output), dim=0)
                    if i == 0:
                        self.y_val_true = torch.cat((self.y_val_true, target), dim=0)
        self.y_val_pred_mean = torch.mean(torch.stack(self.y_val_pred, dim=0), dim=0)
        val_loss = self.loss_fn(self.y_val_pred_mean, self.y_val_true).item()
        return val_loss
    
    def predict(self) -> Tuple:
        """
        Evaluate the ensemble of ICK models on the test data
        """
        y_test_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_test_true = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[TEST]:
                    data, target = self._assign_device_to_data(batch[0], batch[1])
                    output = self.model[i](data).float()
                    y_test_pred[i] = torch.cat((y_test_pred[i], output), dim=0)
                    if i == 0:
                        y_test_true = torch.cat((y_test_true, target), dim=0)
        y_test_pred_mean = torch.mean(torch.stack(y_test_pred, dim=0), dim=0)
        y_test_pred_std = torch.std(torch.stack(y_test_pred, dim=0), dim=0)
        return y_test_pred_mean.detach().cpu().numpy(), y_test_pred_std.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class CMICKEnsembleTrainer(EnsembleTrainer):
    """
    Trainer class for Causal Multi-task Implicit Composite Kernel (CMICK) ensemble
    Note that unlike EnsembleTrainer, all predictors in the ensemble are trained jointly instead of independently

    Arguments
    --------------
    treatment_index: int, the index of the group variable (control or treatment) in the input data
    """
    def __init__(self, model: List, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 num_jobs: int = None, model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = FactualMSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, treatment_index: int = 0, 
                 logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.treatment_index = treatment_index
        stop_criterion = 'loss'
        super(CMICKEnsembleTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, num_jobs, model_save_dir, model_name, 
                                                  loss_fn, device, validation, epochs, patience, verbose, stop_criterion, logger)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        assert self.treatment_index >= 0, "Treatment index must be a non-negative integer."
        assert isinstance(self.loss_fn, (FactualMSELoss, FactualCrossEntropyLoss, FactualMSELoss_MT)), \
            "Loss function must be either FactualMSELoss, FactualMSELoss_MT, or FactualCrossEntropyLoss."
        super(CMICKEnsembleTrainer, self)._validate_inputs()

    def _set_optimizer(self) -> None:
        """
        Only one optimizer is needed for the ensemble
        """
        self.optimizer = OPTIMIZERS[self.optim]([{'params': m.parameters(), **self.optim_params} for m in self.model])
    
    def _train_step(self) -> float:
        """
        Perform a single training step for a baselearner in the CMICK ensemble
        """
        y_train_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_train_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        n_treatments = self.model[0].n_treatments if hasattr(self.model[0], "n_treatments") else 2
        for _, batch in enumerate(self.data_generators[TRAIN]):
            data, target = self._assign_device_to_data(batch[0], batch[1])
            data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
            if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                continue
            predictions = torch.empty(0, data[0].shape[0], n_treatments).to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            for i in range(len(self.model)):
                self.model[i].to(self.device)
                self.model[i].train()
                # Forward pass
                output = self.model[i](data).float()
                predictions = torch.cat((predictions, torch.unsqueeze(output,dim=0)), dim=0)
                # Record the predictions
                y_train_pred[i] = torch.cat((y_train_pred[i], output), dim=0)
            # Backward pass
            loss = self.loss_fn(predictions, target, group)
            loss.backward()
            self.optimizer.step()
            # Record true values and groups
            y_train_true = torch.cat((y_train_true, target), dim=0)
            groups = torch.cat((groups, group), dim=0)
        train_loss = self.loss_fn(torch.stack(y_train_pred), y_train_true, groups).item()
        return train_loss
    
    def train(self) -> None:
        """
        Train the CMICK ensemble
        """
        # initialize the early stopping counter
        best_loss = 1e9
        best_model_state_dict = None
        trigger_times = 0

        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            # Training
            self.loss_fn.regularize_var = True  # Empirical risk-based Bayes
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7.6f}")
            train_start = time.time()
            train_loss = self._train_step()
            # Log the training time and loss
            train_time = time.time() - train_start
            self.logger.info("Training time - {:.0f}s - loss {:.4f}".format(train_time, train_loss))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Validation
            self.loss_fn.regularize_var = False
            val_start = time.time()
            self.logger.info("Validation:")
            val_loss = self.validate()
            val_time = time.time() - val_start
            self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Early stopping
            if val_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best ensemble
                    self.logger.info("Early stopping - patience reached")
                    if best_model_state_dict is not None:
                        self.logger.info("Restoring the best model")
                        for i in range(len(self.model)):
                            self.model[i].load_state_dict(best_model_state_dict['model_'+str(i)])
                    if self.model_save_dir is not None:
                        self.logger.info("Saving the best ensemble")
                        torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                best_loss = val_loss
                best_model_state_dict = {'model_'+str(i): self.model[i].state_dict() for i in range(len(self.model))}
            # Visualize the test predictions if verbose > 0
            if self.verbose > 0:
                with torch.no_grad():
                    y_test_pred_mean, y_test_pred_std, y_test_true = self.predict()
                stats_for_test_pred = self._log_prediction_stats(y_test_pred_mean, y_test_true, y_test_pred_std)
                if self.verbose > 1:
                    self._plot_predictions(y_test_pred_mean, y_test_true, stats_for_test_pred)
        if trigger_times < self.patience:
            self.logger.info("Training completed.")
    
    def validate(self) -> float:
        """
        Evaluate the CMICK ensemble on the validation data
        """
        y_val_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_val_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[key]:
                    data, target = self._assign_device_to_data(batch[0], batch[1])
                    data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
                    if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                        continue
                    output = self.model[i](data).float()
                    y_val_pred[i] = torch.cat((y_val_pred[i], output), dim=0)
                    if i == 0:
                        y_val_true = torch.cat((y_val_true, target), dim=0)
                        groups = torch.cat((groups, group), dim=0)
        y_val_pred = torch.stack(y_val_pred, dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true, groups).item()
        return val_loss
    
    def predict(self, output_binary: bool = False) -> Tuple:
        """
        Evaluate the CMICK ensemble on the test data

        Arguments
        --------------
        output_binary: bool, whether to output binary predictions
        """
        y_test_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_test_true = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[TEST]:
                    data, target = self._assign_device_to_data(batch[0], batch[1])
                    data = data[:self.treatment_index] + data[self.treatment_index+1:]
                    output = self.model[i](data).float()
                    y_test_pred[i] = torch.cat((y_test_pred[i], output), dim=0)
                    if i == 0:
                        y_test_true = torch.cat((y_test_true, target), dim=0)
        y_test_pred_mean = torch.mean(torch.stack(y_test_pred, dim=0), dim=0)
        y_test_pred_std = torch.std(torch.stack(y_test_pred, dim=0), dim=0)
        if output_binary:
            return (y_test_pred_mean > 0.5).float().detach().cpu().numpy(), y_test_pred_std.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()
        else:
            return y_test_pred_mean.detach().cpu().numpy(), y_test_pred_std.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

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