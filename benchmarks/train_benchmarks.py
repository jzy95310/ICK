# train_benchmarks.py: a file containing the definition of training functions for benchmark models
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch

import os
import logging
import time
from typing import Dict, List, Tuple
from utils.train import Trainer, EnsembleTrainer
from utils.losses import CFRLoss, DONUTLoss
from utils.constants import *
from utils.losses import *

class JointNNTrainer(Trainer):
    """
    Class for training the Joint NN benchmark model
    The joint NN model `f` will learn a residual `r = y - scale_factor * y_pred_aug_feature` with the input x, where 
    y_pred_aug_feature is the prediction from an ML model (e.g. random forest, SVM, etc.). 

    Arguments
    --------------
    model: torch.nn.Module, the joint NN model to be trained
    data_generators: Dict, a dict of data generators for the joint NN model where keys must be 'train', 'val', and 'test'
    optim: str, the name of the optimizer to use for training the joint NN model
    optim_params: Dict, a dict of parameters for the optimizer
    lr_scheduler: torch.optim.lr_scheduler, the learning rate scheduler to use for training the joint NN model
    model_save_dir: str, the directory to save the trained joint NN model. If None, the model will not be saved
    model_name: str, the name of the trained joint NN model
    loss_fn: torch.nn.modules.loss._Loss, the loss function for optimizing the joint NN model
    device: torch.device, the device to train the model on
    validation: bool, whether to validate the model during training, default to True
    epochs: int, the number of epochs to train the model for
    patience: int, the number of epochs to wait before early stopping
    verbose: int, the level of verbosity for the trainer, default to 0.
        verbose = 0: no logging
        verbose = 1: log all statistics of test predictions
        verbose = 2: log and plot all statistics of test predictions
    scale_factor: float, the scale factor to be multiplied to the prediction from the ML model
    logger: logging.Logger, an instance of logging.Logger for logging messages, errors, exceptions

    References
    --------------
    [1]. Zheng, Tongshu, et al. "Local PM2. 5 Hotspot Detector at 300 m Resolution: A Random Forest–Convolutional 
    Neural Network Joint Model Jointly Trained on Satellite Images and Meteorology." Remote Sensing 13.7 (2021): 1356.
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), device: torch.device = torch.device('cpu'), 
                 validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, stop_criterion: str = 'loss', 
                 scale_factor: float = 0.95, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.scale_factor = scale_factor
        super(JointNNTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                             loss_fn, device, validation, epochs, patience, verbose, stop_criterion, logger)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        assert self.scale_factor >= 0 and self.scale_factor <= 1, "The scale factor must be between 0 and 1."
        super(JointNNTrainer, self)._validate_inputs()
    
    def _assign_device_to_data(self, batch: Tuple[torch.Tensor]) -> Tuple:
        """
        Assign the device to the data and target
        """
        return tuple([m.to(self.device).float() for m in batch])
    
    def train(self) -> None:
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
                x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward and backward pass
                residual = torch.squeeze(self.model(x, aug_feature)).float()
                y_pred = y_pred_aug_feature * self.scale_factor + residual
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                # Record the predictions
                y_train_pred = torch.cat((y_train_pred, y_pred), dim=0)
                y_train_true = torch.cat((y_train_true, y), dim=0)
            # Log the training time and loss
            train_time = time.time() - train_start
            train_loss = self.loss_fn(y_train_pred, y_train_true).item()
            self.logger.info("{:.0f}s for {} steps - {:.0f}ms/step - loss {:.4f}" \
                  .format(train_time, step + 1, train_time * 1000 // (step + 1), train_loss))
            # Validation
            if self.data_generators[VAL] is not None:
                val_start = time.time()
                self.logger.info("Validation:")
                val_loss = self.validate()
                val_time = time.time() - val_start
                self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            else:
                val_loss = train_loss
            # Early stopping
            if val_loss > best_loss:
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
                best_loss = val_loss
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
        Evaluate the joint NN model on the validation data
        """
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[VAL]:
                x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
                residual = torch.squeeze(self.model(x, aug_feature)).float()
                y_pred = y_pred_aug_feature * self.scale_factor + residual
                y_val_pred = torch.cat((y_val_pred, y_pred), dim=0)
                y_val_true = torch.cat((y_val_true, y), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true).item()
        return val_loss
    
    def predict(self) -> Tuple:
        """
        Evaluate the joint NN model on the test data
        """
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
                residual = torch.squeeze(self.model(x, aug_feature)).float()
                y_pred = y_pred_aug_feature * self.scale_factor + residual
                y_test_pred = torch.cat((y_test_pred, y_pred), dim=0)
                y_test_true = torch.cat((y_test_true, y), dim=0)
        return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class JointNNEnsembleTrainer(EnsembleTrainer):
    """
    Class for training the joint NN ensemble

    Arguments
    --------------
    num_jobs: int, the number of jobs to run in parallel
    """
    def __init__(self, model: List, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 num_jobs: int = None, model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 scale_factor: float = 0.95, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.scale_factor = scale_factor
        super(JointNNEnsembleTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, num_jobs, model_save_dir, model_name, 
                                                     loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        assert self.scale_factor >= 0 and self.scale_factor <= 1, "The scale factor must be between 0 and 1."
        super(JointNNEnsembleTrainer, self)._validate_inputs()
    
    def _assign_device_to_data(self, batch: Tuple[torch.Tensor]) -> Tuple:
        return tuple([m.to(self.device).float() for m in batch])
    
    def _train_step(self, base_learner_idx: int) -> float:
        """
        Perform a single training step for a baselearner in the joint NN ensemble

        Arguments
        --------------
        base_learner_idx: int, the index of the baselearner in the joint NN ensemble
        """
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        self.model[base_learner_idx].train()
        self.model[base_learner_idx].to(self.device)
        for _, batch in enumerate(self.data_generators[TRAIN]):
            x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
            # Zero the gradients
            self.optimizers[base_learner_idx].zero_grad()
            # Forward and backward pass
            residual = torch.squeeze(self.model[base_learner_idx](x, aug_feature)).float()
            y_pred = y_pred_aug_feature * self.scale_factor + residual
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizers[base_learner_idx].step()
            y_train_pred = torch.cat((y_train_pred, y_pred), dim=0)
            y_train_true = torch.cat((y_train_true, y), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true).item()
        return train_loss
    
    def validate(self) -> None:
        """
        Evaluate the joint NN ensemble on the validation data
        """
        y_val_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_val_true = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[VAL]:
                    x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
                    residual = torch.squeeze(self.model[i](x, aug_feature)).float()
                    y_pred = y_pred_aug_feature * self.scale_factor + residual
                    y_val_pred[i] = torch.cat((y_val_pred[i], y_pred), dim=0)
                    if i == 0:
                        y_val_true = torch.cat((y_val_true, y), dim=0)
        y_val_pred_mean = torch.mean(torch.stack(y_val_pred, dim=0), dim=0)
        val_loss = self.loss_fn(y_val_pred_mean, y_val_true).item()
        return val_loss
    
    def predict(self) -> Tuple:
        """
        Evaluate the joint NN ensemble on the test data
        """
        y_test_pred = [torch.empty(0).to(self.device) for _ in range(len(self.model))]
        y_test_true = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(len(self.model)):
                self.model[i].eval()
                for batch in self.data_generators[TEST]:
                    x, aug_feature, y, y_pred_aug_feature = self._assign_device_to_data(batch)
                    residual = torch.squeeze(self.model[i](x, aug_feature)).float()
                    y_pred = y_pred_aug_feature * self.scale_factor + residual
                    y_test_pred[i] = torch.cat((y_test_pred[i], y_pred), dim=0)
                    if i == 0:
                        y_test_true = torch.cat((y_test_true, y), dim=0)
        y_test_pred_mean = torch.mean(torch.stack(y_test_pred, dim=0), dim=0)
        y_test_pred_std = torch.std(torch.stack(y_test_pred, dim=0), dim=0)
        return y_test_pred_mean.detach().cpu().numpy(), y_test_pred_std.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class CFRNetTrainer(Trainer):
    """
    A class for training the Counterfactual Regression Network (CFRNet) proposed by Shalit et al. (2017)

    References
    --------------
    Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment 
    effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017.
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = CFRLoss(), device: torch.device = torch.device('cpu'), 
                 validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, treatment_index: int = 0, 
                 logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.treatment_index = treatment_index
        super(CFRNetTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                            loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        assert self.treatment_index >= 0, "Treatment index must be a non-negative integer."
        assert isinstance(self.loss_fn, CFRLoss), "Loss function must be an instance of CFRLoss."
        super(CFRNetTrainer, self)._validate_inputs()
    
    def _train_step(self) -> Tuple:
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        phi_outputs = torch.empty(0).to(self.device)

        self.model.to(self.device)
        self.model.train()
        for step, batch in enumerate(self.data_generators[TRAIN]):
            data, target = self._assign_device_to_data(batch[0], batch[1])
            data, group = data[:self.treatment_index] + data[self.treatment_index+1:], data[self.treatment_index]
            if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                continue
            data = data[0] if len(data) == 1 else data
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            y_pred, phi_out = self.model(data, group)
            y_pred, phi_out = y_pred.reshape(-1).float(), phi_out.float()
            group, target = group.reshape(-1).float(), target.reshape(-1).float()
            # Backward pass
            loss = self.loss_fn(y_pred, target, group, phi_out)
            loss.backward()
            self.optimizer.step()
            # Record predictions, true values, and groups
            y_train_pred = torch.cat((y_train_pred, y_pred), dim=0)
            y_train_true = torch.cat((y_train_true, target), dim=0)
            groups = torch.cat((groups, group), dim=0)
            phi_outputs = torch.cat((phi_outputs, phi_out), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true, groups, phi_outputs).item()
        return train_loss, step
    
    def validate(self) -> float:
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        phi_outputs = torch.empty(0).to(self.device)
        self.model.eval()

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for batch in self.data_generators[key]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data, group = data[:self.treatment_index] + data[self.treatment_index+1:], data[self.treatment_index]
                if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                    continue
                data = data[0] if len(data) == 1 else data
                y_pred, phi_out = self.model(data, group)
                y_pred, phi_out = y_pred.reshape(-1).float(), phi_out.float()
                group, target = group.reshape(-1).float(), target.reshape(-1).float()
                y_val_pred = torch.cat((y_val_pred, y_pred), dim=0)
                y_val_true = torch.cat((y_val_true, target), dim=0)
                groups = torch.cat((groups, group), dim=0)
                phi_outputs = torch.cat((phi_outputs, phi_out), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true, groups, phi_outputs).item()
        return val_loss
    
    def predict(self) -> Tuple:
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data = data[:self.treatment_index] + data[self.treatment_index+1:]
                data = data[0] if len(data) == 1 else data
                y_pred = self.model.predict(data)
                y_pred, target = y_pred.float(), target.reshape(-1).float()
                y_test_pred = torch.cat((y_test_pred, y_pred), dim=0)
                y_test_true = torch.cat((y_test_true, target), dim=0)
        return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class DCNTrainer(Trainer):
    """
    Trainer class for Deep Counterfactual Network (w/o propensity dropout) proposed by Alaa et al. (2017)

    References
    --------------
    Alaa, Ahmed M., Michael Weisz, and Mihaela Van Der Schaar. "Deep counterfactual 
    networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017).
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = FactualMSELoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 treatment_index: int = 0, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.treatment_index = treatment_index
        super(DCNTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                           loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        assert self.treatment_index >= 0, "Treatment index must be a non-negative integer."
        assert isinstance(self.loss_fn, (FactualMSELoss, FactualCrossEntropyLoss)), "Loss function must be either FactualMSELoss or FactualCrossEntropyLoss."
        super(DCNTrainer, self)._validate_inputs()
    
    def _train_step(self) -> float:
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        self.model.to(self.device)
        self.model.train()
        for step, batch in enumerate(self.data_generators[TRAIN]):
            data, target = self._assign_device_to_data(batch[0], batch[1])
            data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
            if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                continue
            data = data[0] if len(data) == 1 else data
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            output = self.model(data).float().to(self.device)
            loss = self.loss_fn(output, target, group)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Record the predictions
            y_train_pred = torch.cat((y_train_pred, output), dim=0)
            y_train_true = torch.cat((y_train_true, target), dim=0)
            groups = torch.cat((groups, group), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true, groups).item()
        return train_loss, step
    
    def validate(self) -> float:
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        self.model.eval()

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for batch in self.data_generators[key]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
                if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                    continue
                data = data[0] if len(data) == 1 else data
                output = self.model(data).float().to(self.device)
                y_val_pred = torch.cat((y_val_pred, output), dim=0)
                y_val_true = torch.cat((y_val_true, target), dim=0)
                groups = torch.cat((groups, group), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true, groups).item()
        return val_loss
    
    def predict(self) -> Tuple:
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data = data[:self.treatment_index] + data[self.treatment_index+1:]
                data = data[0] if len(data) == 1 else data
                output = self.model.predict(data).float().to(self.device)
                y_test_pred = torch.cat((y_test_pred, output), dim=0)
                y_test_true = torch.cat((y_test_true, target), dim=0)
        return y_test_pred.detach().cpu().numpy(), y_test_true.detach().cpu().numpy()

class DONUTTrainer(Trainer):
    """
    Trainer class for Deep Orthogonal Networks for Unconfounded Treatments (DONUT) as proposed
    by Hatt and Stefan (2021)

    References
    --------------
    Hatt, Tobias, and Stefan Feuerriegel. "Estimating average treatment effects via orthogonal 
    regularization." Proceedings of the 30th ACM International Conference on Information & Knowledge 
    Management. 2021.
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                 model_save_dir: str = None, model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = DONUTLoss(), 
                 device: torch.device = torch.device('cpu'), validation: bool = True, epochs: int = 100, patience: int = 10, verbose: int = 0, 
                 treatment_index: int = 0, logger: logging.Logger = logging.getLogger("Trainer")) -> None:
        self.treatment_index = treatment_index
        super(DONUTTrainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                           loss_fn, device, validation, epochs, patience, verbose, logger)
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        assert self.treatment_index >= 0, "Treatment index must be a non-negative integer."
        assert isinstance(self.loss_fn, DONUTLoss), "Loss function must be an instance of DONUTLoss."
        super(DONUTTrainer, self)._validate_inputs()
    
    def _train_step(self) -> float:
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        global_pred = torch.empty(0).to(self.device)
        self.model.to(self.device)
        self.model.train()
        
        with torch.no_grad():
            for batch in self.data_generators[TRAIN]:
                data, _ = self._assign_device_to_data(batch[0], batch[1])
                data = data[:self.treatment_index] + data[self.treatment_index+1:]
                data = data[0] if len(data) == 1 else data
                output = self.model(data).float().to(self.device)
                global_pred = torch.cat((global_pred, output[:,:2]), dim=0)

        for step, batch in enumerate(self.data_generators[TRAIN]):
            data, target = self._assign_device_to_data(batch[0], batch[1])
            data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
            if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                continue
            data = data[0] if len(data) == 1 else data
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            output = self.model(data).float().to(self.device)
            loss = self.loss_fn(output, target, group, global_pred.detach())
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Record the predictions
            y_train_pred = torch.cat((y_train_pred, output), dim=0)
            y_train_true = torch.cat((y_train_true, target), dim=0)
            groups = torch.cat((groups, group), dim=0)
        train_loss = self.loss_fn(y_train_pred, y_train_true, groups, global_pred.detach()).item()
        return train_loss, step
    
    def validate(self) -> float:
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        groups = torch.empty(0).to(self.device)
        self.model.eval()

        key = TRAIN if not self.validation else (VAL if self.data_generators[VAL] is not None else TEST)
        with torch.no_grad():
            for batch in self.data_generators[key]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data, group = data[:self.treatment_index] + data[self.treatment_index+1:], torch.squeeze(data[self.treatment_index])
                if len(group.shape) == 0 or all(group == 0) or all(group == 1):
                    continue
                data = data[0] if len(data) == 1 else data
                output = self.model(data).float().to(self.device)
                y_val_pred = torch.cat((y_val_pred, output), dim=0)
                y_val_true = torch.cat((y_val_true, target), dim=0)
                groups = torch.cat((groups, group), dim=0)
        val_loss = self.loss_fn(y_val_pred, y_val_true, groups, y_val_pred[:,:2]).item()
        return val_loss
    
    def predict(self) -> Tuple:
        y_test_pred = torch.empty(0).to(self.device)
        y_test_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                data, target = self._assign_device_to_data(batch[0], batch[1])
                data = data[:self.treatment_index] + data[self.treatment_index+1:]
                data = data[0] if len(data) == 1 else data
                output = self.model(data).float().to(self.device)
                y_test_pred = torch.cat((y_test_pred, output[:,:2]), dim=0)
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