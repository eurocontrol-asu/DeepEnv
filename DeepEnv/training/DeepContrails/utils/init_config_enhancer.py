"""
File Name: init_config_enhancer.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: December 2023
Last Modified : December 2023
Description : 

This Python module defines a decorator function called InitConfigEnhancer that can be applied to classes to enhance their initialization process (__init__ method) 
with custom configuration options. It takes one parameter, cls, which is the class to be enhanced. The decorator replaces the original __init__ method of the 
class with an enhanced version, enhanced_init, which takes additional keyword arguments for configuration. If the class has a super_init method, it calls it 
before performing configuration updates. The decorator extracts default keyword arguments from the original __init__ and constructs new keyword arguments based 
on the provided kwargs. It handles a special case where it can import libraries specified in the configuration and attach them to the instance. Finally, it sets
the updated configuration values as instance attributes and calls the original __init__ method with the provided arguments. This decorator can be used to easily 
customize and extend the initialization behavior of classes with additional configurations.
"""

import importlib
import inspect


def InitConfigEnhancer(cls):
    """
    A decorator for enhancing the initialization (__init__) of a class with custom configuration.

    Parameters:
    cls (class): The class to be enhanced.

    Returns:
    class: The enhanced class with modified __init__ method.
    """

    # Storing the original __init__ method of the class
    original_init = cls.__init__

    def enhanced_init(self, *args, **kwargs):
        """
        Enhanced __init__ method that updates the instance with additional configurations.

        Parameters:
        self (instance): An instance of the class.
        kwargs (dict): Keyword arguments for the configuration.
        """
        if hasattr(cls, 'super_init'):
            self.super_init()

        # Getting default keyword arguments from the original __init__
        default_kwargs = inspect.signature(original_init).parameters
        default_keys = default_kwargs.keys()
        
        new_args = [key for key in default_keys if key != 'self' and default_kwargs[key].default == inspect._empty]
        new_args_dict = {key:value for key,value in zip(new_args, args)}
        
        # Constructing new keyword arguments
        new_kwargs = {key: kwargs.get(key, default_kwargs[key].default) 
                      for key in default_keys 
                      if key != 'self' and default_kwargs[key].default != inspect._empty}
        
        if "libs" in new_kwargs:
            libs = new_kwargs.pop("libs")
            for name, value in libs.items():
                lib = importlib.import_module(value) if isinstance(value, str) else value
                setattr(self, name, lib)
        
        # Setting updated configuration values as instance attributes
        for key, value in new_kwargs.items():
            setattr(self, key, value)
            
        for key, value in new_args_dict.items():
            setattr(self, key, value)

        # Calling the original __init__ method
        original_init(self, *args, **kwargs)
    
    
    # Replace the original __init__ with the enhanced one
    cls.__init__ = enhanced_init
    return cls