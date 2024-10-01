"""
File Name: typing_utils.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : December 2023
Description : 

Python module for typing constants

"""

from typing import TypeVar

# Creating type aliases using NewType
np_ndarray = TypeVar('np_ndarray')
torch_tensor = TypeVar('torch_tensor')
torch_nn_module = TypeVar('torch_nn_module')