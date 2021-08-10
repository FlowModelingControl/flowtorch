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
