"""
File Name: image_augmentor.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 


This Python module features the ImageAugmentor class, a powerful tool for image augmentation and preprocessing, 
leveraging the albumentations library. It is designed to support a wide array of image transformations including flipping, 
rotating, and tensor conversions, making it highly versatile for various image processing tasks. The class can handle both 
individual images and batches, applying transformations efficiently. The constructor allows for the use of mock modules, 
facilitating ease of testing with numpy and albumentations. This makes the class adaptable for different environments and 
testing scenarios. The augment_data method, a key component of this class, enables users to apply the defined transformations 
to their data seamlessly, supporting an array of image types and formats. This module is especially useful in machine learning 
and computer vision projects where image data requires sophisticated preprocessing and augmentation to enhance model performance.
"""

from typing import List, Any, Dict

from DeepContrail.utils.init_config_enhancer import InitConfigEnhancer
from DeepContrail.utils.typing_utils import np_ndarray


@InitConfigEnhancer
class ImageAugmentor:
    """
    Class for preprocessing images using albumentations library.

    This class provides a customizable image preprocessing pipeline using the albumentations
    library. It supports the use of mock modules for testing purposes.

    Parameters:
        parameters (Dict[str, Any]): Configuration parameters for the image transformations.
        mock_numpy (Optional[Any]): A mock numpy module for testing purposes. Defaults to None.
        mock_albumentations (Optional[Any]): A mock albumentations module for testing purposes. Defaults to None.
    """
    
    DEFAULT_LIBS = {
        "np": "numpy",
        "albumentations": "albumentations",
        "albumentations.pytorch": "albumentations.pytorch"
    }
       
     
    def __init__(
        self,
        parameters: Dict[str, Any] = None,
        libs: Dict[str, Any] = DEFAULT_LIBS
    ):
        """
        Parameters:
            parameters (Dict[str, Any]): Configuration parameters for the image transformations.
            libs (Dict[str, Any]): A dictionnary of librairies to load. Default to DEFAULT_LIBS.
        """

        # Set up the image transformation pipeline using albumentations
        self.transform = self.albumentations.Compose([
            self.albumentations.HorizontalFlip(p=0.5),
            self.albumentations.VerticalFlip(p=0.5),
            self.albumentations.RandomRotate90(p=0.5),
            self.albumentations.pytorch.ToTensorV2()
        ])
    
    
        self.transform2 = self.albumentations.Compose([
            self.albumentations.HorizontalFlip(p=0.5),
            self.albumentations.VerticalFlip(p=0.5),
            self.albumentations.RandomRotate90(p=0.5),
            self.albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
            self.albumentations.GaussianBlur(blur_limit=(3, 7), p=0.5),
            self.albumentations.pytorch.ToTensorV2()
        ])
    
    def augment_data(self, data: np_ndarray) -> np_ndarray:
        """
        Augment data using predefined transformations.

        Parameters:
            data (np.ndarray): The data to be augmented.

        Returns:
            np.ndarray: The augmented data.
        """

        # Define names for data elements
        data_names = ["image", 'mask', 'weight'][:len(data)]
        
        # Create dictionary of data with corresponding names
        transform_args = {name: value for name, value in zip(data_names, data)}


        if len(data[0].shape) == 3:
            data_transformed = self.transform(**transform_args)
            augmented_data = [data_transformed[name].half() for name in data_names]
        else:
            augmented_data = data
            
        return augmented_data
