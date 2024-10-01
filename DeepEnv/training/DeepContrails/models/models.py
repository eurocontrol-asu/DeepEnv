"""
File Name: typing_utils.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module combines advanced neural network architectures for video data processing and model creation, leveraging 
the capabilities of MONAI, PyTorch, and segmentation_models_pytorch. The primary class, AttentionBasedVideoUNet, integrates 
3D and 2D U-Net models with attention mechanisms, optimizing them for video analysis tasks. Additionally, the module provides 
a create_model_full_channel function that constructs a custom sequential neural network tailored for multi-channel input data. 
This function starts with batch normalization, followed by several convolutional layers, and concludes with an architecture-specific 
model, adaptable to various configurations including Segformer. This module is ideal for researchers and engineers working on advanced
video processing and multi-channel image analysis in deep learning.
"""

import monai
import torch

import segmentation_models_pytorch as smp

import torch
import monai.networks.nets

class AttentionBasedVideoUNet(torch.nn.Module):
    """
    A neural network module that combines 3D and 2D U-Net architectures with attention mechanisms for video data processing.

    This class defines a model architecture that first processes input data with a 3D U-Net
    and then feeds the output into a 2D U-Net. Both U-Nets utilize attention mechanisms to 
    enhance feature extraction in video analysis tasks.

    Attributes:
        model_3d: The 3D U-Net model with attention.
        model_2d: The 2D U-Net model with attention.
    """

    def __init__(self):
        """
        Initializes the AttentionBasedVideoUNet module with 3D and 2D U-Net models.
        """
        super().__init__()

        # Parameters for the 3D U-Net model
        params_3d = {
            "spatial_dims": 3,
            "in_channels": 9,
            "out_channels": 1,
            "channels": (8, 16, 32, 64),
            "strides": (2, 2, 2, 2),
        }

        # Parameters for the 2D U-Net model
        params_2d = {
            "spatial_dims": 2,
            "in_channels": 8,
            "out_channels": 1,
            "channels": (4, 8, 16, 32),
            "strides": (2, 2, 2),
        }

        # Names of the architecture for both 3D and 2D models
        architecture_2d = 'AttentionUnet'
        architecture_3d = 'AttentionUnet'

        # Initialize the 3D and 2D models using MONAI's network architectures
        self.model_3d = getattr(monai.networks.nets, architecture_3d)(**params_3d)
        self.model_2d = getattr(monai.networks.nets, architecture_2d)(**params_2d)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the AttentionBasedVideoUNet model.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the 3D and 2D models.
        """
        # Process the input through the 3D model and then squeeze the output
        output_3d = self.model_3d(x)
        output_3d_squeezed = output_3d.squeeze(1)

        # Process the output of the 3D model through the 2D model
        return self.model_2d(output_3d_squeezed)

    
def create_model_full_channel(self, channel_count: int = 9) -> torch.nn.Module:
    """
    Creates a sequential model with a series of convolutional layers followed by an architecture-specific model.

    This function constructs a neural network model that begins with batch normalization and
    a series of convolutional layers, followed by a specific architecture like Segformer or other
    models from segmentation_models_pytorch (smp). It is designed for processing multi-channel input data.

    Parameters:
        channel_count (int): The number of input channels. Defaults to 9.

    Returns:
        torch.nn.Module: The constructed sequential neural network model.
    """

    # Define batch normalization for the input channels
    batch_norm = torch.nn.BatchNorm2d(channel_count)

    # Define convolutional layers
    conv_layer_1 = torch.nn.Conv2d(channel_count, channel_count, kernel_size=5, stride=1, padding=2)
    conv_layer_2 = torch.nn.Conv2d(channel_count, 6, kernel_size=3, stride=1, padding=1)
    conv_layer_3 = torch.nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)

    # Define the activation function
    sigmoid_activation = torch.nn.Sigmoid()

    # Load a specific model architecture based on the class attribute
    if self.model_architecture == "Segformer":
        # Load a pretrained Segformer and modify its last convolutional layer
        specific_model = Segformer.from_pretrained(self.backbone)
        specific_model.decode_head.classifier = torch.nn.Conv2d(
            specific_model.decode_head.classifier.in_channels,
            1,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
    else:
        # Load other model architecture from segmentation_models_pytorch
        specific_model = getattr(smp, self.model_architecture)(**self.smp_kwargs)

    # Construct the sequential model with the defined layers
    model = torch.nn.Sequential(
        batch_norm,
        conv_layer_1,
        conv_layer_2,
        conv_layer_3,
        sigmoid_activation,
        specific_model
    )
    
    return model
