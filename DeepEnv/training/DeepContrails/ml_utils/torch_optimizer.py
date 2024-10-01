"""
File Name: torch_optimizer.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module introduces the TorchOptimizer class, a utility designed to simplify the usage of various 
optimizers available in PyTorch. The class acts as a centralized repository for common optimizer types, 
defining them as constants such as ADADELTA, ADAGRAD, ADAM, ADAMW, ADAMAX, RMSPROP, and SGD. It includes a class 
method get_optimizer, which dynamically creates an instance of the desired optimizer using its type name. 
This method supports the flexibility of passing additional arguments and keyword arguments, allowing for 
easy customization of optimizer parameters. The TorchOptimizer class enhances the convenience and readability 
of PyTorch-based code by abstracting the complexities involved in instantiating various optimizers, making it 
an essential tool for developers working with PyTorch for machine learning applications.
"""

import torch

from typing import Any, Type

class TorchOptimizer:
    """
    Class for PyTorch optimizers.

    This class defines constants for common optimizer types in PyTorch and provides a method
    to create an optimizer instance based on these types.
    """

    # Defining constants for different types of optimizers
    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    ADAMW = "AdamW"
    ADAMAX = "Adamax"
    RMSPROP = "RMSprop"
    SGD = "SGD"

    @classmethod
    def get_optimizer(cls, optimizer_type: str, *args, **kwargs) -> Type[torch.optim.Optimizer]:
        """
        Returns an instance of the specified PyTorch optimizer.

        This method dynamically retrieves an optimizer class from the `torch.optim` module
        based on the optimizer type name provided. It supports passing additional arguments
        to configure the optimizer.

        Parameters:
            optimizer_type (str): The type of optimizer to create. Should be one of the
                                  class constants like ADADELTA, ADAGRAD, etc.
            *args: Variable length argument list for optimizer parameters.
            **kwargs: Arbitrary keyword arguments for optimizer parameters.

        Returns:
            torch.optim.Optimizer: An instance of the PyTorch optimizer.
        """
        # Dynamically getting the optimizer class from torch.optim and creating an instance
        return getattr(torch.optim, optimizer_type)(*args, **kwargs)
