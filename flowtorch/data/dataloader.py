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
    def get_data_matrix(self, bounds: list = []) -> Tensor:
        """Load data and generate a data matrix.

        For *n* snapshots with *m* datapoints each, this methods returns
        am *m x n* tensor.

        :param bounds: Optional list to load only a subset of the data matrix
        :type bounds: list, optional
        """        
        pass