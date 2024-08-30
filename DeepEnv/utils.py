# @internal
#  @author: Gabriel JARRY
# @endinternal

import numpy as np

def compute_once(function_to_memoize):
    """
    Return a memoïzation of a function that store the result at first apply and the use the stored value
    :param function_to_memoize: is a function to memoïze
    :return: he memoïzed function
    """
    cache = {}

    def wrapper(param):
        if param not in cache:
            cache[param] = function_to_memoize(param)
        return cache[param]

    return wrapper
