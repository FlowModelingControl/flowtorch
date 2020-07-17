r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FunctionDataLoader` allows to generate data based on generic
Python function. Its purpose is mainly to use analytic examples for testing
and validation.
"""

from .dataloader import Dataloader
import torch as pt

class FunctionDataloader(Dataloader):
    r"""Implementation of a concrete :class:`Dataloader` class to generate data based on Python functions.
    """
    
    def __init__(self, function, args):
        r"""Constructor method of :class:`FunctionDataloader` class

        :param function: function to generate data
        :type function: function
        :param args: function arguments
        :type args: list
        """
        self._function = function
        self._args = args

    def _check_bounds(self, bounds) -> bool:
        data_shape = self._data.shape
        valid = True
        if not (len(bounds) == 4):
            valid = False
        elif any(not b.is_integer() for b in bounds):
            valid = False
        elif (bounds[0] < 0) or (bounds[2] < 0):
            valid = False
        elif (bounds[1] > data_shape[0]) or (bounds[3] > data_shape[1]):
            valid = False
        return valid

    def get_data_matrix(self, bounds: list = []) -> pt.Tensor:
        try:
            self._data = pt.as_tensor(self._function(*self._args))
            dim = len(self._data.shape)
            assert dim == 2
        except AssertionError:
            print("Datamatrix has wrong dimension: {:d} (should be 2)".format(dim))
        except Exception as e:
            print("Could not create data matrix: ", e)
        else:
            if self._check_bounds(bounds):
                return self._data[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            else:
                return self._data
        finally:
            print("Finished attempt to create data matrix.")



