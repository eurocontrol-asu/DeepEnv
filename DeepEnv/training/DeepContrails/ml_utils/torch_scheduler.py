"""
Module Name: torch_scheduler
Description: Python module defining torch scheduler class
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
"""

import torch
import transformers

class TorchScheduler:
    """
    Class for PyTorch schedulers.
    """
    
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_ANNEALING_LR = "CosineAnnealingLR"
    COSINE_ANNEALING_WARM_RESTARTS = "CosineAnnealingWarmRestarts"

    @classmethod
    def get_scheduler(cls, scheduler_type: str, *args, **kwargs):
        """
        Returns an instance of the PyTorch scheduler based on the specified scheduler type.

        Parameters:
            scheduler_type (str): The type of scheduler to retrieve
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or transformers.Scheduler: An instance of the PyTorch or transformers scheduler.
        """
        if scheduler_type == cls.COSINE_ANNEALING_LR:
            return torch.optim.lr_scheduler.CosineAnnealingLR(**kwargs)
        elif scheduler_type == cls.REDUCE_LR_ON_PLATEAU: 
            return torch.optim.lr_scheduler.ReduceLROnPlateau(**kwargs)
        elif scheduler_type == cls.COSINE : 
            return transformers.get_cosine_schedule_with_warmup(**kwargs)
        elif scheduler_type in [cls.LINEAR, cls.COSINE_RESTARTS, cls.POLYNOMIAL, cls.CONSTANT, cls.CONSTANT_WARMUP, cls.INVERSE_SQRT, cls.COSINE_ANNEALING_WARM_RESTARTS]:
            return transformers.get_scheduler(scheduler_type, **kwargs)
        else:
            raise RuntimeError("Unknown scheduler type")