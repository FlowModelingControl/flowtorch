r"""Definition of a common interface for all dataloaders.

This abstract base class should be used as parent class when
defining new dataloaders, e.g., for to support additional file
formats.
"""

from abc import ABC, abstractmethod
from torch import Tensor

class Dataloader(ABC):
    r"""Abstract base class to define a common interface for dataloaders.
    """
    @abstractmethod
    def write_times(self):
        pass


    @abstractmethod
    def field_names(self):
        pass


    @abstractmethod
    def load_snapshot(self, time, fields):
        pass


    @abstractmethod
    def load_mesh(self):
        pass


    @abstractmethod
    def weighting_matrix(self):
        pass


    @abstractmethod
    def data_matrix(self, times: list = [], fields: list = []) -> Tensor:
        """Load data and generate a data matrix.

        For *n* snapshots with *m* datapoints each, this methods returns
        am *m x n* tensor.

        :param bounds: Optional list to load only a subset of the data matrix
        :type bounds: list, optional
        """        
        pass