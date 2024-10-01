"""
File Name: torch_generator.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

This Python module contains the TorchGenerator class, a custom data generator extending torch.utils.data.Dataset, designed to 
dynamically generate batches of data for PyTorch models. It is initialized with two callable functions: len_function and get_item_function. 
The len_function is used to determine the total length of the dataset, effectively setting the number of batches per epoch, while the 
get_item_function is responsible for retrieving a specific batch of data based on a given index. This flexible and generic structure 
allows TorchGenerator to adapt to various types of data sources and formats, making it a versatile tool for handling diverse data input 
needs in PyTorch-based machine learning workflows. The use of callable functions for length and item retrieval makes this class highly 
adaptable, enabling it to fit into different data processing pipelines seamlessly.
"""

import torch
from typing import Callable, List

class TorchGenerator(torch.utils.data.Dataset):

    def __init__(self, length_function: Callable[[], int], get_item_function: Callable[[int], List]):
        """
        Initialize the TorchGenerator.

        Parameters:
            length_function (Callable[[], int]): A function that returns the length of the dataset.
            get_item_function (Callable[[int], List]): A function that returns a batch of data given an index.

        The class is initialized with two functions: one for determining the length of the dataset
        and another for retrieving a specific item or batch of data from the dataset.
        """
        self.length_function = length_function
        self.get_item_function = get_item_function

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        This method calls the length_function provided during initialization to determine
        the length of the dataset.

        Returns:
            int: The number of batches per epoch.
        """
        return self.length_function()

    def __getitem__(self, index: int) -> List:
        """
        Generates one batch of data.

        This method retrieves a batch of data corresponding to the given index by calling
        the get_item_function provided during initialization.

        Parameters:
            index (int): The index of the batch.

        Returns:
            List: The batch of data.
        """
        return self.get_item_function(index)

