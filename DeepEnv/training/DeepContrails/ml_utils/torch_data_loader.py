"""
File Name: torch_data_loader.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module, part of the DeepContrail library, provides a comprehensive and versatile data loading 
functionality for deep learning models, specifically tailored for image and video segmentation tasks. The core class, 
TorchDataLoader, is equipped to handle various data formats and modalities, including full images, video sequences, 
and panel data. It supports multiple operational modes like learning, prediction, and stacking, and accommodates a 
range of data types and mask generation methods. The class is highly configurable, allowing adjustments for batch size, 
data shuffling, image resizing, and augmentation. Furthermore, it integrates seamlessly with PyTorch, offering functionalities 
like normalization and tensor conversion. This module is also test-friendly, with support for mock modules, ensuring its 
adaptability in different development and testing environments. The TorchDataLoader is an essential component for efficient 
data preparation in machine learning workflows within the DeepContrail framework, particularly enhancing the model training 
and evaluation processes.
"""

import inspect
import importlib

from typing import Any, Dict, Optional, List, Union, Tuple

from DeepContrail.utils.init_config_enhancer import InitConfigEnhancer
from DeepContrail.utils.typing_utils import np_ndarray, torch_tensor
from DeepContrail.config.default_config import RAW_DIR, GOLD_DIR

@InitConfigEnhancer
class TorchDataLoader: 
    # Class constants for directory names
    FOLDER_VALIDATION = "VALIDATION"
    FOLDER_TRAIN = "TRAIN"
    
    # Class constants for operational modes
    MODE_LEARNING = "LEARNING"
    MODE_PREDICT = "PREDICT"
    MODE_PNG = "PNG"
    MODE_STACK = "STACK"
    MODE_STACK_EVAL = "STACK_EVAL"
    MODE_FILE = "FILE"
    
    # Class constants for data types
    DATA_FULL = "FULL"
    DATA_VIDEO = "VIDEO"
    DATA_VIDEO_FULL = "VIDEO_FULL"
    DATA_PANEL = "PANEL"
    
    # Class constants for mask types
    MASK_VOTE = "VOTE"
    MASK_MEAN = "MEAN"
    MASK_MIN = "MIN"
    MASK_MAX = "MAX"
    
    DEFAULT_LIBS = {
        "os": "os",
        "np": "numpy",
        "torch": "torch",
        "skimage" : "skimage",
        "torchvision": "torchvision",
        "torch_generator": "DeepContrail.ml_utils.torch_generator",
        "image_processing": "DeepContrail.image_utils.image_processing",
        "image_augmentor": "DeepContrail.image_utils.image_augmentor"
    }
    
    def __init__(self, 
                 raw_directory: str = RAW_DIR,
                 gold_directory: str = GOLD_DIR,
                 data_folder: str = "TRAIN",  # Options: TRAIN, VALIDATION, TEST
                 mode: str = "LEARNING", # Options : LEARNING, PREDICT, STACK, STACK_EVAL, FILE
                 batch_size: int = 32,
                 data_type: str = "ASH",  # Options: ASH, FULL, VIDEO, VIDEO_FULL, PANEL
                 data_fold: Optional[float] = None,
                 data_augmentation: Optional[Dict[str, Any]] = None,
                 data_shuffling: bool = True,
                 pixel_shift: bool = False,
                 image_resize: int = 256,
                 image_base_resize: int = 256,
                 image_resize_function: Optional[Any] = None,
                 mask_type: str = "VOTE",  # Options: VOTE, MEAN, MIN, MAX
                 weight: bool = False,
                 weight_dict: Optional[Dict[str, Any]] = None,
                 frame_indexes: List[int] = [i for i in range(8)], 
                 file_list: Optional[List[str]] = None,
                 add_file_list: Optional[List[str]] = None,
                 remove_file_list: Optional[List[str]] = None,
                 data_dict: Optional[Dict[str, Any]] = None, 
                 libs: dict = DEFAULT_LIBS
                ) -> None:
        """
        Initializes the TorchDataLoader with configuration settings.

        Parameters:
            raw_directory (str): Directory path for raw data. Defaults to RAW_DIR.
            gold_directory (str): Directory path for gold (processed) data. Defaults to GOLD_DIR.
            data_folder (str): Data folder of the data loader (e.g., TRAIN, VALIDATION...). Defaults to 'TRAIN'.
            mode (str): Operational mode of the data loader (e.g., LEARNING, PREDICT...). Defaults to 'LEARNING'.
            batch_size (int): Size of the batch for data processing. Defaults to 32.
            data_type (str): Type of data to be processed (e.g., ASH, FULL). Defaults to 'ASH'.
            data_fold (Optional[float]): Fold number for cross-validation, if applicable. Default is None.
            data_augmentation (Optional[Dict[str, Any]]): Parameters for data augmentation. Default is None.
            data_shuffling (bool): Indicates if the data should be shuffled. Defaults to True.
            pixel_shift (bool): Indicates if the data should be shifted from 0.5 pixel (labelling error)
            image_resize (int): Target size for image resizing. Defaults to 256. Default is False
            image_base_resize (int): Base size for image resizing before further processing. Defaults to 256.
            image_resize_function (Optional[Any]): Function identifier for image resizing. Default is None.
            mask_type (str): Type of mask to be used in processing (e.g., MEAN, MIN...). Defaults to 'MEAN'.
            weight (bool): Flag to indicate if weighting should be applied. Defaults to False.
            frame_indexes (List[int]): List of frame indexes to be used. Defaults to the first 8 indexes.
            file_list (Optional[List[str]]): List of files to be included. Default is None.
            add_file_list (Optional[List[str]]): List of additional files to be included. Default is None.
            remove_file_list (Optional[List[str]]): List of files to be excluded. Default is None.
            data_dict (Optional[Dict[str, Any]]): Dictionary of additional data parameters. Default is None.
            weight_dict (Optional[Dict[str, Any]]): Dictionary for weight configuration. Default is None.
            libs (Dict[str, Any]): A dictionnary of librairies to load. Default to DEFAULT_LIBS.
        """
        
        # Initialize file list and image loading function
        
        if file_list is not None:
            self.file_list = file_list
        else:
            self.init_file_list() 
         
        self.load_data_function = self.load_data if self.mode not in [self.MODE_PREDICT, self.MODE_PNG] else self.load_data_predict
        
        # Determine if the last batch should be dropped based on mode
        drop_last_batch = not self.mode in [self.MODE_PREDICT, self.MODE_STACK_EVAL]
        
        # Determine if video data type is being used
        self.is_video_data = self.data_type in [self.DATA_VIDEO, self.DATA_PANEL, self.DATA_VIDEO_FULL]
        
        # Initialize image normalization transform
        self.normalize_image = self.torchvision.transforms.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        # Initialize image preprocessor if data augmentation is enabled
        if self.data_augmentation:
            self.augmentor = self.image_augmentor.ImageAugmentor(parameters=self.data_augmentation)
            
        # Set up data generator and data loader for torch
        self.data_generator = self.torch_generator.TorchGenerator(self.len_samples_per_epoch, self.get_sample)
        self.data_loader = self.torch.utils.data.DataLoader(
            self.data_generator, 
            batch_size=self.batch_size,
            shuffle=self.data_shuffling, 
            pin_memory=True, 
            num_workers=self.os.cpu_count() // 2 - 1,
            drop_last=drop_last_batch
        )
   
    
    def init_file_list(self) -> None:
        """
        Initializes the file list based on the configuration settings.

        This method sets up a list of files to be used by the data loader, 
        depending on various conditions such as data type, data fold, and additional configurations.
        """
        
        # Construct validation file list
        validation_files_path = "/".join([self.raw_directory, self.FOLDER_VALIDATION])
        validation_file_list = ["/".join([self.FOLDER_VALIDATION, filename])  for filename in self.os.listdir(validation_files_path)  if "." not in filename]

        # Construct training file list
        training_files_path = "/".join([self.raw_directory, self.FOLDER_TRAIN])
        training_file_list = ["/".join([self.FOLDER_TRAIN, filename])  for filename in self.os.listdir(training_files_path) if "." not in filename]

        # Adjust frame indexes for DATA_PANEL data type
        if self.data_type == self.DATA_PANEL: 
            self.frame_indexes = [1, 2, 3, 4]

        # Handling custom data dictionary
        if self.data_dict is not None:   
            if isinstance(self.data_fold, list):
                fold, size = self.data_fold
                self.file_list = [el for i, el in enumerate(validation_file_list) if i % size in fold]
                self.data_dict = None
                self.mode = self.MODE_PREDICT
            else:
                self.file_list = ["/".join([self.data_folder, filename]) for filename in self.data_dict.keys()]
        else:
            # Constructing the default file list
            data_files_path = "/".join([self.raw_directory, self.data_folder])
            self.file_list = ["/".join([self.data_folder, filename]) for filename in self.os.listdir(data_files_path) if "." not in filename]
            
            # Filtering file list based on data_fold
            if self.data_fold is not None:
                if isinstance(self.data_fold, list):
                    fold, size = self.data_fold
                    self.file_list = [el for i, el in enumerate(training_file_list) if i % size in fold]
                elif 0.01 <= self.data_fold <= 1.0:
                    num_fold = int(len(self.file_list) * self.data_fold)
                    self.file_list = self.np.random.choice(self.file_list, num_fold)
                elif self.data_fold > 1.0:
                    comparison_operator = self.np.greater if self.mode == self.MODE_PREDICT else self.np.less_equal
                    self.file_list = [el for i, el in enumerate(self.file_list) if comparison_operator(i % 10, self.data_fold - 1)]

        # Adding additional files to the list
        if self.add_file_list is not None:
            self.file_list += self.add_file_list

        # Removing specified files from the list
        if self.remove_file_list is not None:
            self.file_list = [
                el for el in self.file_list 
                if el not in self.remove_file_list
            ]
    
    
    def __call__(self) -> Any:
        """
        Returns the data loader object.

        Returns:
            Any: The data loader object.
        """
        return self.data_loader
    
    
    def len_batches_per_epoch(self) -> int:
        """
        Returns the number of batches per epoch.
        
        Returns:
            int: The number of batches per epoch.
        """
        return len(self.file_list) // self.batch_size
    
    
    
    def on_epoch_end(self) -> None:
        """
        Callback function called at the end of each epoch.
        """
        
        # Shuffle the file list
        if self.data_shufflings:
            self.np.random.shuffle(self.file_list)
    
    
    def len_samples_per_epoch(self) -> int:
        """
        Returns the number of samples per epoch.

        Returns:
            int: The number of samples per epoch.
        """
        return len(self.file_list)
    
    
    def get_batch(self, index : int, file_name=False) -> List: 
        """
        Generates one batch of data.
        
        Parameters:
            index (int): The index of the batch.
        
        Returns:
            List: The batch of data.
        """
        # Calculate the start and end indexes within the buffered images
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Extract the batch of images from the buffer
        batch_images_file = self.file_list[start_index:end_index]
        
        batch_images = [self.load_data_function(file) for file in batch_images_file]

        # Preprocess the batch of images
        preprocessed_batch = self.preprocess_batch(batch_images)
        if file_name:
            return preprocessed_batch, batch_images_file
        else:
            return preprocessed_batch
    
    
    def preprocess_batch(self, batch_images: List) -> Tuple:
        """
        Preprocesses a batch of images.

        Parameters:
            batch_images (list): The batch of images to preprocess.

        Returns:
            tuple: A tuple containing the preprocessed data (preprocessed_x, preprocessed_y, preprocessed_w).
        """
        
        if self.mode in [self.MODE_PREDICT, self.MODE_PNG]:
            return self.torch.stack(batch_images)
        else:
            
            # Extract the x, y, and w values from the batch images
            preprocessed_x = self.torch.stack([image[0] for image in batch_images])
            preprocessed_y = self.torch.stack([image[1] for image in batch_images])

            if self.weight:
                preprocessed_w = self.torch.stack([image[2] for image in batch_images])
                return preprocessed_x, preprocessed_y, preprocessed_w
            else:
                return preprocessed_x, preprocessed_y
        
        
    def get_sample(self, index: int) -> Any:
        """
        Retrieves a sample based on the provided index from the file list.

        Depending on the operational mode of the data loader, this method either 
        returns the image data along with its file name or just the image data.

        Parameters:
            index (int): The index of the sample to retrieve from the file list.

        Returns:
            Any: The loaded image data, and optionally the file name, depending on the mode.
        """
        # Retrieve file name from the file list using the given index
        file_name = self.file_list[index]
        
        # Load image data using the designated image loading function
        image_data = self.load_data_function(file_name)
        
        # Return image data and file name for certain modes, otherwise just image data
        if self.mode in [self.MODE_FILE, self.MODE_PREDICT]:
            return image_data, file_name
        else:
            return image_data
    
    
    def normalize_data(self, data: Union[torch_tensor, List[torch_tensor]]) -> torch_tensor:
        """
        Normalizes the input data, which can be a single image or a list of images.

        The function applies normalization to each image using the 'normalize_image' method. 
        It can handle both individual images and batches of images.

        Parameters:
            data (Union[torch.Tensor, List[torch.Tensor]]): The data to be normalized, either a single image 
                                                        or a list of images.

        Returns:
            torch.Tensor: The normalized data.
        """
        # Normalize a single image
        if len(data.shape) == 3:
            return self.normalize_image(data)
        
        # Normalize a list of images
        else:
            normalized_images = [self.normalize_image(image) for image in data]
            return self.torch.stack(normalized_images)
    
    
    def load_file(self, file_path: str) -> Any:
        """
        Loads data from a file at the given path.

        This function opens the file in read-only binary mode and loads its contents using numpy.

        Parameters:
            file_path (str): The path to the file to be loaded.

        Returns:
            Any: The data loaded from the file.
        """
        # Open the file in read-only binary mode
        file_descriptor = self.os.open(file_path, self.os.O_RDONLY)

        with self.os.fdopen(file_descriptor, 'rb') as file:
            # Load the contents of the file using numpy
            loaded_data = self.np.load(file)
        
        # Return the loaded data
        return loaded_data
    
    
    def load_image(self, record_id: str) -> Any:
        """
        Loads an image or a sequence of images based on the specified record identifier.

        This method handles different scenarios based on the data type and operational mode 
        of the data loader. It loads either a single image or a video sequence.

        Parameters:
            record_id (str): The identifier of the record to load.

        Returns:
            Any: The loaded image or sequence of images.
        """
        # Load full image data
        if self.data_type in [self.DATA_FULL, self.DATA_VIDEO_FULL]:
            image_data = self.image_processing.load_image_full(record_id, self.raw_directory, self.is_video_data)
        
        # Load other types of image data
        else:
            if self.mode == self.MODE_LEARNING:
                file_path = self.os.path.join(self.gold_directory, f"{record_id}_video.npy" if self.is_video_data else f"{record_id}.npy")
                image_data = self.load_file(file_path)
            elif self.mode == self.MODE_PNG:
                image_data = self.skimage.io.imread(self.raw_directory + record_id) / 255
            else:
                image_data = self.image_processing.load_image(record_id, self.raw_directory, self.is_video_data)
        
        # Load specified frame images if data is video  
        if self.is_video_data:
            image_data = image_data[self.frame_indexes, ...]

        # Apply pixel shift if enabled
        if self.pixel_shift:
            image_data = self.image_processing.shift_bottom_right(image_data, is_video=self.is_video_data)

        return image_data
    
            
    def load_mask(self, record_id: str) -> np_ndarray:
        """
        Loads a mask file based on a given record ID and the predefined mask type.

        This function first determines the path for pixel and individual masks. 
        It then loads the appropriate mask file depending on the mask type.

        Parameters:
            record_id (str): The unique identifier for the record.

        Returns:
            np_ndarray: The processed mask data as a NumPy ndarray.
        """
        
        # Define the file paths for pixel and individual masks
        mask_pixel_path = self.os.path.join(self.raw_directory, record_id, 'human_pixel_masks.npy')
        mask_individual_path = self.os.path.join(self.raw_directory, record_id, 'human_individual_masks.npy')
        
        # Load the mask data based on the specified mask type
        if self.mask_type == self.MASK_VOTE:
            try:
                # Attempt to load pixel mask data
                mask_data = self.load_file(mask_pixel_path)
            except FileNotFoundError:
                # If pixel mask not found, load individual mask data
                mask_data = self.load_file(mask_individual_path)
                
                # Compute the mean across the last axis and threshold the mask data
                mask_data = self.np.mean(mask_data, axis=-1)
                mask_data = self.np.where(mask_data >= 0.5, 1, 0).astype(self.np.int32)
        else:
            # Load individual mask data
            mask_data = self.load_file(mask_individual_path)

            if self.mask_type == self.MASK_MEAN:
                # Compute the mean across the last axis for mean mask type
                mask_data = self.np.mean(mask_data, axis=-1)
            else:
                # Compute the sum along the specified axes
                sum_axis = self.np.sum(mask_data, axis=(0, 1, 2))
                
                if self.mask_type == self.MASK_MIN:
                    # Use the mask with the minimum sum for min mask type
                    mask_data = mask_data[..., self.np.argmin(sum_axis)]
                elif self.mask_type == self.MASK_MAX:
                    # Use the mask with the maximum sum for max mask type
                    mask_data = mask_data[..., self.np.argmax(sum_axis)]

        return mask_data
    
    
    def load_data_predict(self, record_id: str) -> torch_tensor:
        """
        Loads and preprocesses the image data for prediction based on the record identifier.

        This method handles the loading and preprocessing of image data, including resizing and normalization, 
        specifically for prediction purposes.

        Parameters:
            record_id (str): The identifier of the record to load.

        Returns:
            torch.Tensor: The preprocessed image data ready for prediction.
        """
        # Load image data
        image_data = self.load_image(record_id)
        image_data = self.torch.from_numpy(image_data)
        
        # Rearrange axis based on whether it's video data
        image_data = image_data.moveaxis(-1, 1 if self.is_video_data else 0)
        
        # Resize image data if necessary
        if self.image_resize != self.image_base_resize:
            image_data = self.image_resize_function(image_data)
            
  
        # Normalize image data if not dealing with full data type
        if self.data_type != self.DATA_FULL:
            image_data = self.normalize_data(image_data)
            
            
        # Rearrange axis for video data
        if self.is_video_data: 
            image_data = image_data.moveaxis(1, 0)

        return image_data
    
    
    def load_data(self, record_id: str) -> List[Any]:
        """
        Loads and processes data based on the given record identifier for training or evaluation.

        This method handles the loading of image and mask data and applies additional processing 
        such as augmentation, normalization, and resizing as configured.

        Parameters:
            record_id (str): The identifier of the record to load.

        Returns:
            List[Any]: A list containing the processed image and mask data, along with optional weights.
        """
        # Handle special case for stack modes
        if self.mode in [self.MODE_STACK, self.MODE_STACK_EVAL]:
            processed_record_id = record_id.split("/")[1]
            return self.data_dict[processed_record_id]
        
        # Load image and mask data
        image_data = self.load_image(record_id)
        mask_data = self.load_mask(record_id)
        
        # Initialize data list with image and mask
        processed_data = [image_data, mask_data]

        # Append weights to data list if specified
        if isinstance(self.weight, self.np.ndarray) or self.weight:
            weight_data = self.weight
            processed_data.append(weight_data)
            
        # Append weights from weight dictionary if available
        if self.weight_dict is not None:
            weight_data = self.weight_dict[record_id]
            processed_data.append(weight_data)
        
        # Apply data augmentation or convert to torch tensors and adjust axes
        if self.data_augmentation: 
            processed_data = self.augmentor.augment_data(processed_data)
        else:
            processed_data = [self.torch.from_numpy(element).half() for element in processed_data]
            processed_data[0] = processed_data[0].moveaxis(-1, 1 if self.is_video_data else 0)
        
        # Adjust mask data axes
        processed_data[1] = processed_data[1].moveaxis(-1, 0)  
        
        # Resize image data if necessary
        if self.image_resize != self.image_base_resize:
            processed_data[0] = self.image_resize_function(processed_data[0])
 
        # Normalize image data if not dealing with full data type
        if self.data_type != self.DATA_FULL:
            processed_data[0] = self.normalize_data(processed_data[0])
                 
        # Rearrange axis for video data
        if self.is_video_data: 
            processed_data[0] = processed_data[0].moveaxis(1, 0)
            
        # Concatenate images for panel data type
        if self.data_type == self.DATA_PANEL:
            processed_data[0] = image_processing.concat_panel_images(processed_data[0])
        
        return processed_data
    
    
