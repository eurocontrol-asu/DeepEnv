"""
File Name: lighting_model.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module provides a comprehensive implementation of a machine learning model using 
the PyTorch Lightning framework. It includes the LightningModel class, which integrates various 
functionalities such as model configuration, training and validation steps, and optimizer configuration. 
Key features include handling of different data types, incorporation of Exponential Moving Average (EMA), 
and utilization of custom loss functions and metrics. The module is designed with flexibility in mind, 
supporting different model architectures, learning rates, and optimizer types. It also offers the capability 
to dynamically load modules for testing purposes. This robust implementation is ideal for advanced 
machine learning applications requiring customizable and efficient training pipelines.
"""

import inspect
import importlib

from typing import List
from DeepContrail.utils.init_config_enhancer import InitConfigEnhancer
from DeepContrail.utils.typing_utils import torch_nn_module, torch_tensor

from typing import Tuple, Callable, List, Any, Dict, Optional, Union

pytorch_lightning = importlib.import_module("pytorch_lightning")

@InitConfigEnhancer
class LightningModel(pytorch_lightning.LightningModule):
    
    DEFAULT_LIBS = {
        "gc": "gc",
        "np": "numpy",
        "torch": "torch",
        "model_ema": "timm.utils.model_ema",
        "torch_scheduler": "DeepContrail.ml_utils.torch_scheduler",
        "torch_optimizer": "DeepContrail.ml_utils.torch_optimizer"
    }

    def super_init(cls):
        super().__init__()
    
    
    def __init__(self,  
                 model: torch_nn_module=None,
                 learning_rate: float = 3e-4,
                 loss: Callable = None,
                 metrics: Callable = None,
                 encoder_freeze: bool = True,
                 image_resize : int = 256,
                 image_base_resize: int = 256,
                 logits: bool = True,
                 scheduler_type : str = "cosine", 
                 optimizer_type : str = "Adam",
                 optimizer_args : dict = {},
                 scheduler_args : dict = {},
                 ema_bool = False,
                 data_type: str = "ASH",
                 libs: dict = DEFAULT_LIBS) -> None:
        """
        Class initializer for the the LightningModel.
        
        It sets up the model based on given arguments. It initializes various training and validation metrics, 
        and optionally sets up Exponential Moving Average (EMA) for the model.

        Parameters:
            model (torch.nn.Module): odel to be trained.
            learning_rate (float): Learning rate for the optimizer.
            loss (Callable): Loss function for model training.
            metrics (Callable): Metrics for evaluating the model performance.
            encoder_freeze (bool): Flag to freeze the encoder in the model. Defaults to True.
            image_resize (int): Target size for image resizing. Defaults to 256.
            image_base_resize (int): Base size for initial image resizing. Defaults to 256.
            logits (bool): Flag to use logits in the output layer. Defaults to True.
            scheduler_type (str): Type of learning rate scheduler. Defaults to "cosine".
            optimizer_type (str): Type of optimizer. Defaults to "Adam".
            optimizer_args (Dict[str, Any]): Additional parameters for the optimizer. Defaults to an empty dictionary.
            scheduler_args (Dict[str, Any]): Additional parameters for the scheduler. Defaults to an empty dictionary.
            ema_bool (bool): Flag to enable Exponential Moving Average. Defaults to False.
            data_type (str): Type of data being processed. Defaults to "ASH".
            libs (Dict[str, Any]): A dictionnary of librairies to load. Default to DEFAULT_LIBS.
        """
        
        # Freeze the encoder parameters if specified in the configuration
        if self.encoder_freeze:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad = False
        
        # Initialize Exponential Moving Average (EMA) if enabled in the configuration
        if self.ema_bool:
            self.ema_model = self.model_ema.ModelEmaV2(self.neural_network_model, decay=1e-3)
                
        # Initialize lists for tracking training and validation metrics
        self.train_batch_loss = []
        self.train_loss = []
        self.train_batch_metrics = [[] for _ in self.metrics]
        self.train_metrics = [[] for _ in self.metrics]

        self.val_batch_loss = []
        self.val_loss = []
        self.val_batch_metrics = [[] for _ in self.metrics]
        self.val_metrics = [[] for _ in self.metrics]
        
        # Initialize counters and lists for validation steps and metrics
        self.epoch_count = 0
        self.val_step_outputs = []
        self.val_step_labels = []
        
        self.thresholds = self.np.arange(0.01, 1.00, 0.01)
        self.val_epoch_nums = {thr: 0 for thr in  self.thresholds}
        self.val_epoch_denums = {thr: 0 for thr in  self.thresholds}
        self.global_val_dices = []
        self.best_thresholds = []


    def forward(self, input_images: torch_tensor, evaluate_ema: Optional[bool] = False) -> torch_tensor:
        """
        Forward pass of the model to generate predictions.

        This method processes the input images through either the primary model or the EMA model,
        depending on the 'evaluate_ema' flag. It also handles resizing and cropping of the output
        based on model configuration.

        Parameters:
            input_images (torch.tensor): The input images to be processed.
            evaluate_ema (Optional[bool]): Flag to indicate whether to use the EMA model for prediction.
                                           Defaults to False.

        Returns:
            Tensor: The output predictions from the model.
        """
        # Use the EMA model for prediction if evaluate_ema is True, otherwise use the primary model
        if evaluate_ema:
            predictions = self.ema_model.module(input_images)
        else:
            predictions = self.model(input_images)
        
        # Resize predictions if required
        if self.image_resize != self.image_base_resize:
            predictions = self.torch.nn.functional.interpolate(predictions, size=self.image_base_resize, mode='bilinear')
        
        # Crop predictions for 'PANEL' data type
        if self.data_type == "PANEL":
            predictions = predictions[..., self.image_resize:, self.image_resize:]
        
        return predictions
        
        

    def training_step(self, batch: Tuple[torch_tensor, torch_tensor], batch_idx: int) -> torch_tensor:
        """
        The training step for the model during each batch of data.

        This method processes a batch of input data and targets, computes the loss, and updates
        the Exponential Moving Average (EMA) model if enabled. It also logs the training loss and 
        computes metrics for monitoring.

        Parameters:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the current batch.
        """

        # Free up unused memory
        self.gc.collect()

        # Extract input data and labels from the batch
        if len(batch) == 2:
            inputs, targets = batch
            logits = self.forward(inputs)
            loss = self.loss(logits, targets)
        else:
            inputs, targets, weights = batch
            logits = self.forward(inputs)
            loss = self.loss(logits, targets, weights)
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store the loss value for the current batch
        self.train_batch_loss.append(loss.item())
        
        # Compute and store metrics for each metric function
        for i, metric in enumerate(self.metrics):
            if self.logits:
                self.train_batch_metrics[i].append(metric(preds=self.torch.sigmoid(logits), target=targets.long()).item())
            else:
                self.train_batch_metrics[i].append(metric(preds=logits, target=targets.long()).item())

        # Update the EMA model if enabled
        if self.ema_bool: 
            self.ema.update(self.model)

        return loss


    def validation_step(self, batch: Tuple[torch_tensor, torch_tensor], batch_idx: int) -> torch_tensor:
        """
        The validation step for the model during each batch of validation data.

        This method processes a batch of validation data, computes the loss, and evaluates
        the performance using defined metrics. It supports processing with or without weights
        and also evaluates using the Exponential Moving Average (EMA) model if enabled.

        Parameters:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch in the validation process.

        Returns:
            torch.Tensor: The computed loss for the current validation batch.
        """

        # Extract input data and labels from the batch
        if len(batch) == 2:
            inputs, targets = batch
            logits = self.forward(inputs, evaluate_ema=self.ema_bool)
            loss = self.loss(logits, targets)
        else:
            inputs, targets, weights = batch
            logits = self.forward(inputs, evaluate_ema=self.ema_bool)
            loss = self.loss(logits, targets, weights)

        # Store the loss value for the current validation batch
        self.val_batch_loss.append(loss.item())
        
        # Compute and store metrics for each metric function during validation
        for i, metric in enumerate(self.metrics):
            if self.logits:
                self.val_batch_metrics[i].append(metric(preds=self.torch.sigmoid(logits), target=targets.long()).item())
            else:
                self.val_batch_metrics[i].append(metric(preds=logits, target=targets.long()).item())
        
        logits = self.torch.sigmoid(logits) if self.logits else logits
        y_t = targets.long()
        
        
        for thr in self.thresholds:
            y_p = self.torch.where(logits > thr, 1.0, 0.0).squeeze()
            intersection = self.torch.sum(y_p * y_t)
            union = self.torch.sum(y_p) + self.torch.sum(y_t)
            self.val_epoch_nums[thr] +=  2*intersection.cpu().numpy()
            self.val_epoch_denums[thr] +=  union.cpu().numpy()
        
        # Store the outputs and labels for further evaluation or logging
        # self.val_step_outputs.append(logits.cpu())
        # self.val_step_labels.append(targets.cpu())

        return loss
    
    def store_mean_values(
        self, 
        loss: List[float], 
        batch_loss: List[float], 
        metrics: List[List[float]], 
        batch_metrics: List[List[float]]
    ):
        """
        Stores the mean values of loss and metrics for each training/validation epoch.

        This method calculates and stores the mean loss and mean values of each metric
        over the batches processed in an epoch. It resets the batch-specific lists after
        computing the means.

        Parameters:
            loss_list (List[float]): The list to store the mean loss for each epoch.
            batch_loss_list (List[float]): The list of loss values for each batch in an epoch.
            metrics_list (List[List[float]]): The list to store the mean values of each metric for each epoch.
            batch_metrics_list (List[List[float]]): The list of metric values for each batch in an epoch.
        """
        # Calculate and append the mean loss value
        loss.append(self.np.mean(batch_loss))

        # Reset the batch loss list
        batch_loss = []

        # Iterate over each metric and its corresponding batch metric values
        for i, batch_metric in enumerate(batch_metrics):
            # Calculate and append the mean metric value
            metrics[i].append(self.np.mean(batch_metric))

            # Reset the batch metric list
            batch_metrics[i] = []
        

    def on_validation_epoch_end(self) -> Dict[str, float]:
        """
        Actions to be performed at the end of each validation epoch.

        This method calculates and logs the mean validation loss and metrics. It also computes 
        a specific metric (e.g., Dice coefficient) for all validation predictions and updates 
        the global validation metric list. It clears the lists of validation outputs and labels 
        after processing.

        Returns:
            Dict[str, float]: A dictionary containing the global validation metric.
        """

        # Store and calculate mean values of loss and metrics for validation
        self.store_mean_values(
            self.val_loss, self.val_batch_loss, 
            self.val_metrics, self.val_batch_metrics
        )

        # Log the epoch number and validation results
        print(f"Epoch {self.epoch_count}")
        print(f"Validation loss: {self.val_loss[-1]}, Validation metric: {self.val_metrics[0][-1]}")

        # Concatenate all predictions and labels from the validation steps
        #all_preds = self.torch.cat(self.val_step_outputs)
        #all_labels = self.torch.cat(self.val_step_labels)

        # Apply sigmoid function if logits are used
        #if self.logits:
        #    all_preds = self.torch.sigmoid(all_preds.double())

        # Clear validation step outputs and labels for the next epoch
        #self.val_step_outputs.clear()
        #self.val_step_labels.clear()

        # Calculate the Dice coefficient for all validation predictions
        global_val_dice, best_thr = self.threshold_dice()
        self.best_thresholds.append(best_thr)
        self.global_val_dices.append(global_val_dice)
        
        self.val_epoch_nums = {thr: 0 for thr in  self.thresholds}
        self.val_epoch_denums = {thr: 0 for thr in  self.thresholds}

        # Log validation loss and global validation Dice coefficient
        self.log("val_loss", self.val_loss[-1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for i, metric in enumerate(self.metrics):
             self.log("val_"+metric.__name__, self.val_metrics[i][-1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("global_val_dice", global_val_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("global_val_dice", global_val_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Print the global validation Dice coefficient
        print("Global validation Dice:", global_val_dice, "Best threshold:", best_thr)

        # Return the global validation Dice coefficient
        return {"global_val_dice": global_val_dice}
    
    
    def on_train_epoch_end(self) -> None:
        """
        Actions to be performed at the end of each training epoch.

        This method calculates and stores the mean training loss and metrics. It also increments 
        the epoch count and logs the training results.
        """

        # Store and calculate mean values of loss and metrics for training
        self.store_mean_values(
            self.train_loss, self.train_batch_loss, 
            self.train_metrics, self.train_batch_metrics
        )

        # Increment the epoch count
        self.epoch_count += 1

        # Log the training results for the epoch
        print(f"Train loss: {self.train_loss[-1]}, Train metric: {self.train_metrics[0][-1]}")


    def predict_step(self, 
                     batch: Tuple[torch_tensor, Any], 
                     batch_idx: int) -> Tuple[torch_tensor, Any]:
        """
        Performs a prediction step for a given batch of data.

        This method applies the model to the input data (x) and returns the prediction logits.
        If Exponential Moving Average (EMA) is enabled, it evaluates the model using EMA parameters.

        Parameters:
            batch (Tuple[torch.Tensor, Any]): A batch of input data. The batch consists of tensors and any 
                                              other type of data, like labels or metadata, not used in prediction.
            batch_idx (int): The index of the batch.

        Returns:
            Tuple[torch.Tensor, Any]: The predicted logits for the given batch.
        """

        # Extract the input data (x) from the batch
        x = batch

        # Compute logits using the forward method, with EMA evaluation if enabled
        logits = self.forward(x, evaluate_ema=self.ema_bool)

        # Return the logits as prediction results
        return logits
    

    def configure_optimizers(self) -> Dict[str, Union[Any, Dict]]:
        """
        Configures the optimizers and learning rate schedulers for training.

        This method sets up the optimizer and scheduler based on the model's configuration. 
        It supports different types of schedulers, including those that adjust the learning rate 
        based on specific metrics.

        Returns:
            Dict[str, Union[torch.optim.Optimizer, Dict]]: A dictionary containing the optimizer and 
                                                          the learning rate scheduler configuration.
        """

        # Create an optimizer with the specified type and parameters
        optimizer = self.torch_optimizer.TorchOptimizer.get_optimizer(self.optimizer_type, 
                                                                      params=self.parameters(), 
                                                                      lr=self.learning_rate, 
                                                                      **self.optimizer_args)

        # Create a scheduler with the specified type and parameters
        scheduler = self.torch_scheduler.TorchScheduler.get_scheduler(self.scheduler_type, 
                                                                      optimizer=optimizer, 
                                                                      **self.scheduler_args)

        # Configure the scheduler dictionary based on its type
        if self.scheduler_type == self.torch_scheduler.TorchScheduler.REDUCE_LR_ON_PLATEAU:
            lr_scheduler_dict = {
                "scheduler": scheduler, 
                "interval": "epoch", 
                "monitor": "global_val_dice"
            }
        else:
            lr_scheduler_dict = {
                "scheduler": scheduler, 
                "interval": "step"
            }

        # Return a dictionary with optimizer and scheduler configurations
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler_dict
        }
    

    def threshold_dice(self) -> float:
        """
        Computes the best dice score for predictions by varying the threshold.

        This method iterates over a range of threshold values to determine the one that provides 
        the best dice score for the given predictions compared to the ground truths.

        Returns:
            float: The best dice score achieved with the optimal threshold.
        """

        # Range of threshold values to test
        best_dice_score = 0.0  # Initialize the best dice score to zero
        best_thr = self.thresholds[0]

        # Iterate over each threshold value to find the best dice score
        for thr in self.thresholds:
            # Compute the dice score for the current threshold
            current_score =  self.val_epoch_nums[thr] / self.val_epoch_denums[thr]

            # Update the best score if the current score is higher
            if current_score > best_dice_score:
                best_dice_score = current_score
                best_thr = thr
                
        # Return the best dice score found
        return best_dice_score, best_thr
        