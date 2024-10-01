"""
File Name: torch_segmentation_trainer.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module, part of the DeepContrail library, is designed for training, evaluating, and managing machine learning models, 
particularly focusing on segmentation tasks in deep learning. The core class, TorchSegmentationTrainer, facilitates the creation, training, 
and evaluation of segmentation models using PyTorch and PyTorch Lightning. It offers extensive configurability, allowing the user to specify 
details like model architecture, backbone, learning rate, data augmentation, and various model-specific parameters. Additionally, it supports 
advanced functionalities like exponential moving averages (EMA), batch normalization in decoders, and various types of data resizing techniques. 
The module also includes methods for plotting training metrics, calibration curves, threshold analysis, and model evaluation, providing a 
comprehensive toolkit for model management and experimentation. The integration of mock modules suggests its suitability for both production 
and testing environments.
"""

import dill
import shutil
import inspect
import importlib
from typing import Any, Dict, Optional, List, Union, Tuple, Callable

from DeepContrail.utils.init_config_enhancer import InitConfigEnhancer
from DeepContrail.utils.typing_utils import np_ndarray
from DeepContrail.config.default_config import RAW_DIR, GOLD_DIR, MODEL_DIR, EVALUATION_DIR, AVAILABLE_ARCHITECTURES, AVAILABLE_BACKBONES
from DeepContrail.loss.loss import dice_loss, dice_coefficient

@InitConfigEnhancer
class TorchSegmentationTrainer:

    DEFAULT_LIBS = {
        "os": "os",
        "ttach": "ttach",
        "numpy": "numpy",
        "torch": "torch",
        "pandas": "pandas",
        "datetime": "datetime",
        "matplotlib": "matplotlib",
        "torchvision": "torchvision",
        "torchsummary":"torchsummary",
        "pytorch_lightning": "pytorch_lightning",
        "calibration": "sklearn.calibration",
        "linear_model": "sklearn.linear_model",
        "models": "DeepContrail.models.models",
        "lightning_model": "DeepContrail.models.lightning_model",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
        "torch_data_loader": "DeepContrail.ml_utils.torch_data_loader"
    }

    
    def __init__(self, 
                raw_directory: str = RAW_DIR,
                gold_directory: str = GOLD_DIR,
                model_directory: str = MODEL_DIR,
                evaluation_directory: str = EVALUATION_DIR,
                loss: Union[str, Callable] = dice_loss,
                metrics: List[Union[str, Callable]] = [dice_coefficient],
                monitor: str = "global_val_dice",
                monitor_mode: str = "max",
                learning_rate: float = 3e-4,
                trainer_args: dict = {},
                model_backbone: str = 'efficientnet-b0',
                model_architecture: str = 'Unet',
                encoder_freeze: bool = False,
                decoder_attention_type: Optional[str] = None,
                decoder_use_batchnorm: bool = True,
                ema_bool: bool = False,
                logits: bool = True,
                batch_size: int = 32,
                accumulate_grad_batches: int = 1,
                image_resize: int = 256,
                image_base_resize: int = 256,
                weight: bool = False,
                weight_dict: Optional[Dict[str, Any]] = None,
                data_shuffling: bool = True,
                frame_indexes: List[int] = [i for i in range(8)], 
                data_type: str = "ASH",  # Options: ASH, FULL, VIDEO, PANEL
                data_fold: Optional[float] = None,
                data_augmentation: Optional[Dict[str, Any]] = None,
                mask_type: str = "VOTE",  # Options: VOTE, MEAN, MIN, MAX
                super_resolution: Optional[str] = None,
                scheduler_type: str = "cosine",
                optimizer_type: str = "Adam",
                scheduler_args: dict = {},
                optimizer_args: dict = {},   
                callbacks: list = [],
                file_list: Optional[list] = None,
                add_file_list: Optional[List[str]] = None,
                remove_file_list: Optional[List[str]] = None,
                model_name_suffix: str = '',
                gpu_indexes: str = '0',
                libs: dict = DEFAULT_LIBS) -> None:
        """
        Initializes the TorchSegmentationTrainer with the given configuration.

        Parameters:
            raw_directory (str): Directory for raw data.
            gold_directory (str): Directory for gold standard data.
            model_directory (str): Directory to save the model.
            evaluation_directory (str): Directory for evaluation outputs.
            loss (Union[str, Callable]): Loss function or its string identifier.
            metrics (List[Union[str, Callable]]): List of metrics or their string identifiers for model evaluation.
            monitor (str): Name of the metric to be monitored during training.
            monitor_mode (str): Mode of monitoring ('max' or 'min').
            learning_rate (float): Learning rate for the optimizer.
            trainer_args (dict): Additional arguments for the trainer.
            model_backbone (str): Backbone model for the segmentation network.
            model_architecture (str): Architecture of the segmentation model.
            encoder_freeze (bool): Whether to freeze the encoder part of the model.
            decoder_attention_type (Optional[str]): Type of attention in the decoder, if any.
            decoder_use_batchnorm (bool): Whether to use batch normalization in the decoder.
            ema_bool (bool): Whether to use Exponential Moving Average.
            logits (bool): Whether to use logits at the output.
            batch_size (int): Batch size for training and evaluation.
            accumulate_grad_batches (int): Number of batches over which to accumulate gradients.
            image_resize (int): Size to which input images should be resized.
            image_base_resize (int): Base size for image resizing.
            weight (bool): Whether to use class weights in loss calculation.
            weight_dict (Optional[Dict[str, Any]]): Dictionary of weights for different classes.
            data_shuffling (bool): Whether to shuffle data during training.
            frame_indexes (List[int]): Indexes of frames to use from video data.
            data_type (str): Type of data to be used (e.g., 'ASH', 'FULL').
            data_fold (Optional[float]): Proportion of data to use for training/validation.
            data_augmentation (Optional[Dict[str, Any]]): Specifications for data augmentation.
            mask_type (str): Method to generate masks ('VOTE', 'MEAN', etc.).
            super_resolution (Optional[str]): Super-resolution method, if any.
            scheduler_type (str): Type of learning rate scheduler.
            optimizer_type (str): Type of optimizer.
            scheduler_args (dict): Additional arguments for the scheduler.
            optimizer_args (dict): Additional arguments for the optimizer.
            callbacks (list): List of callbacks to be used during training.
            file_list (Optional[list]): List of files to be used for training.
            add_file_list (Optional[List[str]]): List of additional files to include.
            remove_file_list (Optional[List[str]]): List of files to exclude from training.
            model_name_suffix (str): Suffix for the model name.
            gpu_indexes (str): Indexes of GPUs to be used for training.
            libs (Dict[str, Any]): A dictionnary of librairies to load. Default to DEFAULT_LIBS.
        Raises:
            RuntimeError: If the model architecture or backbone is unknown.
        """
        self.libs = libs
        
        # Check if the chosen architecture is available
        if self.model_architecture not in AVAILABLE_ARCHITECTURES:
            raise RuntimeError("Unknown segmentation architecture.")
        
        # Check if the chosen backbone model is available
        if self.model_backbone not in AVAILABLE_BACKBONES:
            raise RuntimeError("Unknown backbone model.")
        
        # Set precision for matrix multiplication in torch
        self.torch.set_float32_matmul_precision('medium')
        
        # Set the CUDA visible devices environment variable
        self.os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_indexes
        
        # Determine the names of the loss function and metrics
        self.loss_name = self.loss.__name__ if isinstance(self.loss, Callable) else self.loss
        self.metrics_name = [metric if isinstance(metric, str) else metric.__name__ for metric in self.metrics]
        
        # Set the model name based on the configuration
        self.set_model_name()
        
        # Define the directory path for saving the model
        self.model_dir_path = self.model_directory + self.model_name + "/"
        
        # Print the model name
        print(self.model_name)
        
        # Create the model directory if it doesn't exist
        self.os.makedirs(self.model_dir_path, exist_ok=True)
        
        # Create data loaders for training and validation
        self.create_data_loaders()
        
        # Initialize the model
        self.create_model()
        
        # Initialize a DataFrame to track training history
        self.history_df = self.pandas.DataFrame()
        
    def set_model_name(self) -> None:
        """
        Constructs and sets the model name based on various configuration parameters.
        The name is built using parameters like architecture, backbone, data type, etc.
        """
        
        # List to accumulate parts of the model name
        name_parameters = [self.model_architecture, self.model_backbone]

        # Append specific flags to the name based on the configuration
        if self.decoder_attention_type is not None:
            name_parameters.append("attention")
        if self.encoder_freeze:
            name_parameters.append("frozen")
        if self.decoder_use_batchnorm:
            name_parameters.append("decoder_batchnorm")
        if isinstance(self.weight, self.numpy.ndarray) or self.weight:
            name_parameters.append("weighted")
        name_parameters.append(self.data_type)
        if self.ema_bool: 
            name_parameters.append("ema")

        # Handle image resizing and append the relevant information to the name
        if self.image_resize != self.image_base_resize:
            resize_identifier = "resized_" + str(self.image_resize)
            if self.super_resolution == "fft":
                resize_identifier = self.super_resolution + "_" + str(self.image_resize)
                self.image_resize_function: Callable = lambda img: resize_fft(img, self.image_resize)
            elif self.super_resolution == "wavelet": 
                resize_identifier = self.super_resolution + "_" + str(self.image_resize)
                self.image_resize_function: Callable = lambda img: torch.from_numpy(
                    resize_wavelet(img.moveaxis(1 if self.video else 0, -1).numpy(), self.image_resize)
                ).moveaxis(-1, 1 if self.video else 0).float()
            else:
                self.image_resize_function: Callable = self.torchvision.transforms.transforms.Resize(self.image_resize, antialias=True)
            name_parameters.append(resize_identifier)
        else:
            self.image_resize_function: Callable = lambda el: el

        # Append data fold information if available
        if self.data_fold is not None:
            fold_info = "fold_" + ("".join([str(el) for el in self.data_fold]) if isinstance(self.data_fold, list) else str(self.data_fold))
            name_parameters.append(fold_info)

        # Append other configuration parameters to the name
        name_parameters += [
            self.mask_type, 
            self.optimizer_type, 
            self.scheduler_type, 
            "lr_" + str(self.learning_rate), 
            "bs_" + str(self.batch_size)
        ]

        # Append gradient accumulation info if it's greater than 1
        if self.accumulate_grad_batches > 1:
            name_parameters.append("ga_" + str(self.accumulate_grad_batches))

        # Append the current date and time to the name
        name_parameters.append(str(self.datetime.datetime.now())[:-7].replace(":", ""))

        # Append model name suffix if available
        name_suffix = "-" + self.model_name_suffix if self.model_name_suffix else ""
        self.model_name = "-".join(name_parameters) + name_suffix


    def create_data_loaders(self) -> None:
        """
        Creates data loaders for training and validation datasets.

        This method initializes data loaders with the configuration parameters 
        set for the trainer, taking into account specific configurations for 
        training and validation datasets, such as data shuffling and augmentation.
        """

        # Creating the training data loader with the specified configurations
        self.train_data_loader = self.torch_data_loader.TorchDataLoader(
            raw_directory=self.raw_directory,
            gold_directory=self.gold_directory,
            batch_size=self.batch_size,
            weight=self.weight,
            weight_dict=self.weight_dict,
            data_type=self.data_type,
            data_shuffling=self.data_shuffling,
            data_augmentation=self.data_augmentation,
            mask_type=self.mask_type,
            image_resize=self.image_resize,
            image_base_resize=self.image_base_resize,
            data_fold=[self.data_fold[0], len(self.data_fold[0]) + 1] if isinstance(self.data_fold, list) else self.data_fold,
            data_folder="TRAIN",
            image_resize_function=self.image_resize_function,
            frame_indexes=self.frame_indexes,
            mode="LEARNING",
            file_list=self.file_list,
            add_file_list=self.add_file_list,
            remove_file_list=self.remove_file_list,
        )
        
        # Creating the validation data loader with similar configurations, but without data shuffling and augmentation
        self.val_data_loader = self.torch_data_loader.TorchDataLoader(
            raw_directory=self.raw_directory,
            gold_directory=self.gold_directory,
            batch_size=self.batch_size,
            weight=self.weight,
            weight_dict=self.weight_dict,
            data_type=self.data_type,
            data_shuffling=False,  # No shuffling for validation data
            data_augmentation=False,  # No augmentation for validation data
            mask_type="VOTE",  # Using 'VOTE' mask type for validation
            image_resize=self.image_resize,
            image_base_resize=self.image_base_resize,
            data_fold=[self.data_fold[1], len(self.data_fold[1]) + 1] if isinstance(self.data_fold, list) else self.data_fold,
            data_folder="VALIDATION",
            image_resize_function=self.image_resize_function,
            frame_indexes=self.frame_indexes,
            mode="LEARNING",
            file_list=self.file_list,
            add_file_list=self.add_file_list,
            remove_file_list=self.remove_file_list,
        )
    
    
    def create_model(self) -> None:
        """
        Creates and sets the model for the trainer based on the specified architecture, backbone, and other parameters.

        The method selects the appropriate model creation strategy based on the data type and model architecture.
        It initializes the model with specific parameters and sets it as an attribute of the trainer class.

        Raises:
            RuntimeError: If the data type 'VIDEO' is selected, as it is not yet implemented.
        """

        # Dictionary of keyword arguments for segmentation model creation
        smp_kwargs = {
            "encoder_name": self.model_backbone, 
            "encoder_weights": 'imagenet',
            "decoder_attention_type": self.decoder_attention_type,
            "decoder_use_batchnorm": self.decoder_use_batchnorm,
            "img_size": self.image_resize,
        }
        
        self.smp_kwargs = smp_kwargs

        # Adjusting arguments for specific architectures
        if self.model_architecture in ['DeepLabV3', "DeepLabV3Plus"]:
            smp_kwargs.pop("decoder_attention_type", None)
            smp_kwargs.pop("decoder_use_batchnorm", None)
            
        # Creating model based on the data type
        if self.data_type == "FULL":
            model = self.models.create_model_full_channel(self)
        elif self.data_type == "VIDEO_FULL":
            model = self.models.AttentionBasedVideoUNet()
        elif self.data_type == "VIDEO":
            raise RuntimeError("Not implemented yet")
        else:
            # Creating a standard model from segmentation models pytorch
            model = getattr(self.segmentation_models_pytorch, self.model_architecture)(**smp_kwargs)
        
        model.to(self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu"))
        self.torchsummary.summary(model, input_size=(9 if self.data_type == "FULL" else 3, self.image_resize, self.image_resize))
        
        # Wrapping the model with additional functionality using LightningModel
        self.model = self.lightning_model.LightningModel(
            model=model,
            learning_rate=self.learning_rate,
            image_resize=self.image_resize, 
            image_base_resize=self.image_base_resize,
            loss=self.loss, 
            logits=self.logits,
            metrics=self.metrics, 
            encoder_freeze=self.encoder_freeze, 
            scheduler_type=self.scheduler_type, 
            optimizer_type=self.optimizer_type, 
            scheduler_args=self.scheduler_args,
            optimizer_args=self.optimizer_args,
            ema_bool=self.ema_bool,
            data_type=self.data_type
        )
        

        
    def create_checkpoint(self) -> None:
        """
        Creates a checkpoint for the model using PyTorch Lightning's ModelCheckpoint callback.

        This method initializes a checkpointing mechanism to save the model based on the specified monitoring criteria.
        The best model will be saved based on the performance metric being monitored.
        """

        # Print the directory path where the model will be saved
        print(self.model_dir_path)

        # Create a checkpoint object with PyTorch Lightning
        # This will save the best model based on the specified monitoring metric and mode
        self.checkpoint = self.pytorch_lightning.callbacks.ModelCheckpoint(
            save_top_k=1,               # Save only the best model
            monitor=self.monitor,       # Metric to monitor for performance
            mode=self.monitor_mode,     # Mode for monitoring ('max' or 'min')
            dirpath=self.model_dir_path,  # Directory path to save the model
            filename="model",           # Filename for the saved model
            verbose=True                # Enable verbose output
        )

        
    def fit(self,
            epochs: int = 100,
            verbose: int = 2, 
            accelerator: str = "gpu"
           ) -> None:
        """
        Fits the model to the training data.

        Parameters:
            epochs (int): The number of epochs to train the model.
            verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            accelerator (str): The type of accelerator to use for training (e.g., 'gpu', 'cpu').

        The method initializes a PyTorch Lightning trainer, sets up checkpointing, and 
        trains the model on the provided data. It also compiles the training and validation 
        results into a pandas DataFrame.
        """

        # Create a checkpoint to save the model based on performance
        self.create_checkpoint()

        # Set the progress bar status based on the verbose level
        enable_progress_bar = verbose not in [0, 2]
        
        print("epochs", epochs)

        # Initialize the PyTorch Lightning Trainer with specified configurations
        self.trainer = self.pytorch_lightning.Trainer(
            accelerator=accelerator,
            max_epochs=epochs,
            callbacks=[self.checkpoint] + list(self.callbacks),
            enable_progress_bar=enable_progress_bar,
            precision="16-mixed",
            accumulate_grad_batches=self.accumulate_grad_batches,
            #gradient_clip_val=0.5
        )

        # Train the model using the trainer
        self.trainer.fit(
            self.model, 
            train_dataloaders=self.train_data_loader(), 
            val_dataloaders=self.val_data_loader(),
        )
        
        # Compile the training and validation results
        result_dict = {
            "loss": self.model.train_loss,
            "val_loss": self.model.val_loss[1:]
        }
        
        # Add metrics to the results dictionary
        for train_metric, val_metric, name in zip(self.model.train_metrics, self.model.val_metrics, self.metrics_name):
            result_dict[name] = train_metric
            result_dict["val_" + name] = val_metric[1:]
        
        # Add global dice coefficient to the results
        result_dict["global_dice_coeff"] = self.model.global_val_dices[1:]
        
        result_dict["best_threshold"] = self.model.best_thresholds[1:]
         
        # Create a DataFrame from the results dictionary
        self.current_history_df = self.pandas.DataFrame(result_dict)

    
    def predict(self, 
                samples: Any, 
                threshold: bool = True, 
                proba: bool = True, 
                video: bool = False, 
                calibrated: bool = False) -> np_ndarray:
        """
        Makes predictions on the given samples using the trained model.

        Parameters:
            samples (Any): Input data for making predictions.
            threshold (bool): Whether to apply a threshold to the prediction output.
            proba (bool): Whether to return probabilities.
            video (bool): Whether the input samples are video data.
            calibrated (bool): Whether to use calibration coefficients for prediction.

        Returns:
            np_ndarray: The predicted masks or probabilities.

        Raises:
            RuntimeError: If the model has not been created or loaded.
        """

        # Check if the model has been initialized
        if self.model is None:
            raise RuntimeError("Model has not been created.")

        # Move the model to the appropriate device
        self.model.to(self.device)

        # Make predictions without tracking gradients
        with self.torch.no_grad():
            # Move samples to the appropriate device
            samples = samples.to(self.device)

            # Handling prediction for video data
            if video:
                masks = []
                # Process each frame in the video data
                for i in range(8):
                    masks.append(self.model.forward(samples[:, :, i].float()))
                # Concatenate masks along the axis
                masks = self.torch.cat(masks, axis=1)
            else:
                # Process for non-video data
                masks = self.model.forward(samples.float())

            # Move samples back to CPU
            samples = samples.cpu()

        # Apply calibration if required
        if calibrated:
            a, b = self.calibration_coefficients
            masks = a[0][0] * masks + b[0]

        # Apply sigmoid if returning probabilities or if threshold is enabled and using logits
        if (proba or threshold) and self.logits:
            masks = self.torch.sigmoid(masks)

        # Apply thresholding if required
        if threshold:
            masks = self.torch.where(masks > self.threshold, 1, 0)

        return masks


    def load_model(self, path: str = None) -> None:
        """
        Load a pre-trained model from a file.

        Parameters:
            path (str): The path to the model file (default: None). If None, the default path is used.
        """
        # Determine the model path
        model_path = self.model_directory + self.model_name + "/model.ckpt" if path is None else path
        
        # Load the model
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(self.torch.load(model_path, map_location=self.device)['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        
    
    def data_resize_function(self, data: np_ndarray) -> np_ndarray:
        """
        Resizes the input data using the predefined image resize function.

        This function handles both single images and batches of images. 
        It applies the image resize function set in the trainer to each image.

        Parameters:
            data (np_ndarray): The data to be resized. Can be a single image or a batch of images.

        Returns:
            np_ndarray: The resized data, either as a single image or a batch of images.
        """

        # Check if the input data is a single image (3D array)
        if len(data.shape) == 3:
            # Apply the image resize function to a single image
            resized_data = self.image_resize_function(data)
            return resized_data
        else:
            # Apply the image resize function to each image in a batch (list of images)
            resized_data = []
            for image in data:
                # Resize each image individually
                resized_image = self.image_resize_function(image)
                resized_data.append(resized_image)
            # Stack the resized images into a batch
            return self.torch.stack(resized_data)

        
    def plot_segmentation_results(self, batch_number: int = 1) -> None:
        """
        Plot the segmentation results for current model on a number of validation batch.
        
        Parameters:
        batch_number (int): Number of batches to plot (default: 3).
        """
        # Iter on batch_number
        for step in range(batch_number):
            # Get a batch of samples from the image generator
            batch = self.val_data_loader.get_batch(0)
            batch_x =  batch[0]  
            batch_y = batch[1]

            # Get the predictions for the batch using the segmentation model
            batch_pred = self.predict(batch_x)

            # Plot the sample image, predicted mask, and ground truth mask for each sample in the batch
            fig, axs = self.matplotlib.pyplot.subplots(self.batch_size, 3, figsize=(15, 5*self.batch_size))  # Create subplots

            for i in range(self.batch_size):
                # Plot the sample image
                x = batch_x[i].moveaxis(0, -1)
                y_pred = batch_pred[i].moveaxis( 0, -1)
                y_true = batch_y[i].moveaxis(0, -1)
                    
                axs[i, 0].imshow(x)
                axs[i, 0].axis('off')
                axs[i, 0].set_title('Sample Image')

                # Plot the predicted mask
                axs[i, 1].imshow(y_pred)
                axs[i, 1].axis('off')
                axs[i, 1].set_title('Predicted Mask')

                # Plot the ground truth mask
                axs[i, 2].imshow(y_true)
                axs[i, 2].axis('off')
                axs[i, 2].set_title('Ground Truth Mask')

            # Adjust spacing between subplots
            self.matplotlib.pyplot.tight_layout()

            # Show the plot
            self.matplotlib.pyplot.show()
            
    
    def plot_training_metrics(self) -> None:
        """
        Plots the training metrics including loss and other specified metrics.

        This function creates a series of plots for each metric recorded during training, 
        displaying both training and validation metrics across epochs. The plots are saved 
        as an image file in the model directory.
        """

        # Number of metrics to plot
        num_metrics = len(self.metrics_name)

        # Create subplots for each metric plus the loss
        fig, axs = self.matplotlib.pyplot.subplots(1, 1 + num_metrics, figsize=(5 + num_metrics * 5, 5), dpi=200)

        # Plot training and validation loss
        axs[0].plot(self.history_df["loss"], label='Train')
        axs[0].plot(self.history_df["val_loss"], label='Validation')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel(self.loss_name.replace("_", " ").capitalize())
        axs[0].set_ylim(-0.01, 1.01)
        axs[0].legend()

        # Iterate through each metric and plot
        for i, metric_name in enumerate(self.metrics_name):
            # Special handling for global dice coefficient
            try:
                if "dice_coeff" in metric_name:
                    axs[1 + i].plot(self.history_df["global_dice_coeff"], label='Global')
            except KeyError:
                # Skip if the metric is not implemented
                pass

            # Plot training and validation values for the current metric
            axs[1 + i].plot(self.history_df[metric_name], label='Train')
            axs[1 + i].plot(self.history_df["val_" + metric_name], label='Validation')
            axs[1 + i].set_xlabel('Epochs')
            axs[1 + i].set_ylabel(metric_name.replace("_", " ").capitalize())
            axs[1 + i].set_ylim(-0.01, 1.01)
            axs[1 + i].legend()

        # Adjust layout and save the plot
        self.matplotlib.pyplot.tight_layout()
        self.matplotlib.pyplot.savefig(self.model_dir_path + "train_curves.png")

    
    def write_history(self) -> None:
        """
        Writes the training history to a CSV file.

        This function concatenates the current training history DataFrame with the 
        overall history DataFrame and saves the updated history to a CSV file in the 
        model directory.
        """

        # Concatenate the current history DataFrame with the overall history DataFrame
        self.history_df = self.pandas.concat([self.history_df, self.current_history_df])

        # Save the updated history DataFrame to a CSV file in the model directory
        self.history_df.to_csv(self.os.path.join(self.model_dir_path, "history.csv"), index=False)

    
    def train(self, epochs: int = 100, verbose: int = 2, accelerator: str = "gpu") -> None:
        """
        Trains the model with the specified parameters.

        This method encompasses the entire training process, including creating the model 
        (if not already created), fitting the model to the data, writing the training history, 
        plotting training metrics, and evaluating the model.

        Parameters:
            epochs (int): The number of epochs to train the model.
            verbose (int): The verbosity level of the training process.
            accelerator (str): The type of accelerator to use (e.g., 'gpu' or 'cpu').
        """

        # Check if the model exists, if not, create it
        if self.model is None:
            self.create_model()

        # Fit the model with the specified parameters
        self.fit(epochs=epochs, verbose=verbose, accelerator=accelerator)

        # Write the training history to a file
        self.write_history()

        # Plot the training metrics and save the plots
        self.plot_training_metrics()

        # Evaluate the model's performance
        self.evaluate_model()

    
    def plot_threshold_curve(self, 
                             predictions: np_ndarray, 
                             ground_truths: np_ndarray, 
                             metric_index: int, 
                             not_local: bool = True) -> None:
        """
        Plots a threshold curve for the specified metric.

        This function calculates the performance metric at different threshold values 
        and plots these scores against the thresholds. It also identifies and stores 
        the best threshold value based on the metric.

        Parameters:
            predictions (np_ndarray): The predictions made by the model.
            ground_truths (np_ndarray): The ground truth labels.
            metric_index (int): The index of the metric in the metrics list to be evaluated.
            not_local (bool): If True, the plot is saved as a file; otherwise, it's not saved.
        """

        # Range of threshold values to evaluate
        thresholds = self.numpy.arange(0.01, 1.00, 0.01)

        # List to store scores for each threshold
        scores = []
        self.best_score = 0.0

        # Evaluate the metric at each threshold
        for threshold in thresholds:

            # Calculate the score at the current threshold
            score = self.metrics[metric_index](preds=predictions, target=ground_truths.long(), threshold=threshold)
            score = score.cpu().numpy()

            # Update the best score and threshold if this score is better
            if score > self.best_score:
                self.best_score = score
                self.threshold = threshold
            print(threshold, score)
            scores.append(score)
            self.torch.cuda.empty_cache()

        # Plotting the threshold curve
        self.matplotlib.pyplot.figure(dpi=200)
        self.matplotlib.pyplot.plot(thresholds, scores)
        self.matplotlib.pyplot.xlabel("Threshold value")
        self.matplotlib.pyplot.ylabel(self.metrics_name[metric_index].replace("_", " ").capitalize())
        self.matplotlib.pyplot.xlim(-0.1, 1.1)
        self.matplotlib.pyplot.tight_layout()
        
        # Save the plot if not_local is True
        if not_local:
            self.matplotlib.pyplot.savefig(self.model_dir_path + "threshold_curve.png")

    
    def dump_instance(self, file_name: str) -> None:
        """
        Dumps the current instance of the class to a file.

        This function is used for serialization of the class instance. It nullifies the 
        data loaders to prevent issues with serialization and then serializes the instance 
        using dill. The serialized object is saved to a specified file.

        Parameters:
            file_name (str): The name of the file where the instance will be dumped.
        """
        
        # Open the file with write and create modes
        file_descriptor = self.os.open(file_name, self.os.O_WRONLY | self.os.O_CREAT)

        # Nullify the data loaders to avoid serialization issues
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.model = None
        self.trainer = None

        # Serialize and save the instance using dill
        with self.os.fdopen(file_descriptor, 'wb') as file:
            for lib in self.libs:
                setattr(self, lib, None)
            dill.dump(self, file)

    
    def calibration_plots(self, X: np_ndarray, y: np_ndarray, logreg, save_path: str) -> None:
        """
        Generates and saves calibration plots and histograms.

        This function creates calibration plots to compare the model's predictions 
        before and after calibration with logistic regression. It also generates 
        histograms of the predicted probabilities.

        Parameters:
            X (np_ndarray): The predicted logits from the model.
            y (np_ndarray): The ground truth labels.
            logreg: Logistic regression model used for calibration.
            save_path (str): Directory path where the plots will be saved.
        """

        # Clear the current figure
        self.matplotlib.pyplot.clf()

        # Convert logits to probabilities using sigmoid function
        X_probas = self.torch.nn.Sigmoid()(X)

        # Obtain calibrated probabilities from logistic regression
        predictions = logreg.predict_proba(X)[:, 1]

        # Calculate calibration curve before logistic regression calibration
        prob_true, prob_pred = self.calibration.calibration_curve(y, X_probas, n_bins=20)
        self.matplotlib.pyplot.plot(prob_pred, prob_true, label="Avant calibration")

        # Plot for perfect calibration
        self.matplotlib.pyplot.plot([0, 1], [0, 1], "-.", label="Calibration parfaite")

        # Calculate calibration curve after logistic regression calibration
        prob_true, prob_pred = self.calibration.calibration_curve(y, predictions, n_bins=20)
        self.matplotlib.pyplot.plot(prob_pred, prob_true, label="Aprés calibration")

        # Setup the plot titles and labels
        self.matplotlib.pyplot.title("Calibration plot")
        self.matplotlib.pyplot.xlabel("Mean predicted probability")
        self.matplotlib.pyplot.ylabel("Fraction of positives")
        self.matplotlib.pyplot.legend()

        # Save the calibration plot
        self.matplotlib.pyplot.savefig(self.os.path.join(save_path, "calibration_plot.png"))

        # Clear the current figure
        self.matplotlib.pyplot.clf()

        # Create histograms for the predicted probabilities
        fig, (ax1, ax2) = self.matplotlib.pyplot.subplots(1, 2)
        ax1.hist(predictions[100000:], range=(0, 1), bins=100, label="Avant calibration", log=True)
        ax1.set_title("Aprés calibration")
        ax2.hist(X_probas[100000:].squeeze(1), range=(0, 1), bins=100, log=True)
        ax2.set_title("Avant calibration")

        # Adjust layout and save the histograms
        self.matplotlib.pyplot.tight_layout()
        self.matplotlib.pyplot.savefig(self.os.path.join(save_path, "calibration_histograms.png"))

        # Clear the current figure for future plots
        self.matplotlib.pyplot.clf()

    
    def calibrate(self, logits: np_ndarray, labels: np_ndarray) -> None:
        """
        Calibrates the model using logistic regression.

        This function applies logistic regression to the flattened logits and labels 
        to compute calibration coefficients. It then generates calibration plots 
        based on the logistic regression model.
        
        The function randomly samples a subset of the data for calibration to 
        avoid memory issues and then fits a logistic regression model. The 
        calibration coefficients are stored, and calibration plots are generated.

        Parameters:
            logits (np_ndarray): Logits output from the model.
            labels (np_ndarray): Ground truth labels correspself.torch.cuda.empty_cache()onding to the logits.

        """

        # Flatten logits and labels
        X = logits.flatten()
        y = labels.flatten()

        # Randomly sample a subset of the data
        indexes = self.numpy.random.randint(0, X.shape[0], 10000000)
        X, y = X[indexes].reshape(-1, 1), y[indexes]

        # Fit logistic regression model
        logreg = self.linear_model.LogisticRegression()
        logreg.fit(X, y)

        # Store calibration coefficients
        a, b = logreg.coef_, logreg.intercept_
        self.calibration_coefficients = (a, b)

        # Generate and save calibration plots
        self.calibration_plots(X, y, logreg, self.model_dir_path)

    
    def evaluate_model(self, metric_index: int = 0, path: Optional[str] = None, not_local: bool = True) -> None:
        """
        Evaluates the trained model on the validation dataset.
        
        This method performs model evaluation by generating predictions on the validation dataset,
        optionally applies data augmentation, and performs model calibration. It also plots the
        threshold curve and possibly renames the model directory based on the evaluation score.

        Parameters:
            metric_index (int): Index of the metric in the metrics list to be used for evaluation.
            path (Optional[str]): Path to load the model from, if applicable.
            not_local (bool): If True, additional operations such as model loading, calibration, and instance dumping are performed.
        """

        # Load the model from the given path if not_local is True
        if not_local:
            self.load_model(path=path)
        
        # Apply data augmentation if enabled
        if self.data_augmentation:
            self.model.model = self.ttach.SegmentationTTAWrapper(self.model.model, self.ttach.aliases.d4_transform(), merge_mode='mean')

        # Lists to store predictions, logits, and ground truths
        predictions, logits, ground_truths = [], [], []
        
        # Process each batch in the validation data loader
        i = 0
        for batch in self.val_data_loader():
            # Unpack the batch
            x, y, *_ = batch
            print(i, end="\r")
            x = x.to(self.device).float()
            y = y.to(self.device).float()

            # Generate predictions and logits
            y_logits = self.predict(x, threshold=False, proba=False)
            y_hat = self.torch.sigmoid(y_logits) if self.logits else y_logits

            # Append results to respective lists
            logits.append(y_logits.cpu())
            predictions.append(y_hat.cpu())
            ground_truths.append(y.cpu())
            i += 1
            self.torch.cuda.empty_cache()

        # Concatenate the results across all batches
        predictions = self.torch.cat(predictions, dim=0)
        ground_truths = self.torch.cat(ground_truths, dim=0)
        logits = self.torch.cat(logits, dim=0)
        
        # Reset the model if data augmentation was applied
        if self.data_augmentation:
            self.model.model = self.model.model.model

        # Perform calibration and plot threshold curve if not_local is True
        if not_local:
            self.calibrate(logits.cpu(), ground_truths.cpu())
            self.plot_threshold_curve(predictions.cpu(), ground_truths.cpu(), metric_index, not_local=not_local)
        
        # Print and potentially update model directory based on the evaluation score
        score = self.numpy.round(self.best_score, 4)
        if not_local:
            path_split = self.model_dir_path.split("/")
            path_split[-2] = str(score) + '-' + path_split[-2]
            new_dir_path = "/".join(path_split)
            self.os.rename(self.model_dir_path, new_dir_path)
            file_name = new_dir_path + '/instance.pkl'

            # Dump the instance and move the directory
            self.dump_instance(file_name)
            print(file_name)
            shutil.move(new_dir_path, self.evaluation_directory + "/" + path_split[-2])


        
        
    
            
            
      
        

    