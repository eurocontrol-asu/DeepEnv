"""
File Name: loss.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module offers a set of specialized functions for calculating loss and metrics in image segmentation tasks, 
particularly focused on binary segmentation. It includes dice_loss, a function that computes the Dice loss, 
commonly used for quantifying the similarity between two sets of data in binary image segmentation. 
The dice_coefficient function calculates the Dice coefficient, a widely-used metric for the evaluation of binary 
segmentation model performance. Additionally, the module features the focal_tversky_temperature function, which implements 
the Focal Tversky loss with temperature scaling. This function is particularly useful for adjusting the relative importance 
of false positives and false negatives in segmentation tasks, enhancing model performance by modulating prediction sharpness. 
The module relies on the segmentation_models_pytorch library and torchmetrics for efficient and accurate metric computations, 
making it a valuable tool for developers working on binary image segmentation with deep learning models.
"""


from typing import Any, Dict
from DeepContrail.utils.typing_utils import torch_tensor

import segmentation_models_pytorch as smp
from torchmetrics.functional import dice
import torch



def dice_loss(*args: Any) -> Any:
    """
    Computes the Dice loss for the given inputs using segmentation_models_pytorch.

    Dice loss is used for measuring the similarity between two sets of data, which is widely used 
    for binary image segmentation tasks. This implementation uses a smoothing factor and is set to 
    binary mode, making it suitable for binary segmentation tasks.

    Parameters:
        *args (Any): Variable length argument list. Typically includes the predicted outputs and ground truth labels.

    Returns:
        Any: The computed Dice loss based on the provided arguments.
    """
    # Calculate Dice loss using segmentation_models_pytorch with binary mode and smoothing
    return smp.losses.DiceLoss(mode="binary", smooth=1)(*args)


def dice_coefficient(*args: Any, **kwargs: Dict[Any, Any]) -> Any:
    """
    Calculates the Dice coefficient, a measure of overlap between two samples.

    This function is a wrapper around the Dice metric provided by torchmetrics. It is suitable for 
    evaluating the performance of binary segmentation tasks.

    Parameters:
        *args (Any): Variable length argument list, typically including predictions and target labels.
        **kwargs (Dict[Any, Any]): Keyword arguments passed to the Dice metric function.

    Returns:
        Any: The computed Dice coefficient based on the provided arguments.
    """
    # Calculate Dice coefficient using torchmetrics
    return dice(*args, **kwargs)


def focal_tversky_temperature(y_pred: torch_tensor, 
                              y_true: torch_tensor, 
                              temperature: float = 1.0, 
                              alpha: float = 0.7, 
                              beta: float = 0.3, 
                              gamma: float = 4/3) -> torch_tensor:
    """
    Applies the Focal Tversky loss function with a temperature adjustment to the predicted and true values.

    The Tversky index is an asymmetric similarity measure that allows for the adjustment of the relative 
    importance between false negatives and false positives. The temperature scaling adjusts the sharpness 
    of the predictions.

    Parameters:
        y_pred (torch.Tensor): The predicted values. 
        y_true (torch.Tensor): The ground truth values.
        temperature (float): The temperature scaling factor. Default is 1.0.
        alpha (float): The weight for false positives. Default is 0.7.
        beta (float): The weight for false negatives. Default is 0.3.
        gamma (float): The focusing parameter of the Tversky index. Default is 4/3.

    Returns:
        torch.Tensor: The computed Focal Tversky loss.
    """

    # Apply temperature scaling to the predictions
    scaled_predictions = y_pred / temperature

    # Calculate and return the Focal Tversky loss
    tversky_loss = smp.losses.TverskyLoss("binary", alpha=alpha, beta=beta, gamma=1/gamma, from_logits=True)
    return tversky_loss(y_pred=scaled_predictions, y_true=y_true)



def wbce_loss(logits, labels, pos_weight=5) :
    pos_weight = torch.Tensor([pos_weight]).cuda()
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(input = logits, target = labels.float())


def hybrid_loss(logits, labels, pos_weight=5):
    return dice_loss(logits, labels) + wbce_loss(logits, labels, pos_weight=pos_weight)


