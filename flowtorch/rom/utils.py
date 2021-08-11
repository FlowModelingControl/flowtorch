"""Collection of utilities for reduced-order modeling.
"""

# standars library packages
from time import time
import functools
from typing import Callable


def log_time(func) -> dict:
    """Measure and log a function's execution time.

    :param func: function to be executed; the function is expected
        to return a dictionary
    :type func: Callable
    :return: dictionary returned by the wrapped function with additional
        entry for execution time
    :rtype: dict
    """
    @functools.wraps(func)
    def measure_time(*args, **kwargs) -> dict:
        start_time = time()
        log = func(*args, **kwargs)
        return {**log, "execution_time": time()-start_time}
    return measure_time


def check_positive_integer(value: int, name: str):
    """Check if an input value is a positive integer.

    :param value: input value to check
    :type value: int
    :param name: name of the parameter
    :type name: str
    :raises ValueError: if the argument is not an integer
    :raises ValueError: if the argument is less than one
    """
    message = f"The argument of {name} must be a positive integer"
    if not isinstance(value, int):
        raise ValueError(message)
    if value < 1:
        raise ValueError(message)
