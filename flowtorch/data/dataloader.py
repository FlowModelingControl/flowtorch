"""Definition of a common interface for all dataloaders.

This abstract base class should be used as parent class when
defining new dataloaders, e.g., to support additional file
formats.
"""

from abc import ABC, abstractmethod
from torch import Tensor

BIG_INT = 1e15


class Dataloader(ABC):
    r"""Abstract base class to define a common interface for dataloaders.
    """
    @abstractmethod
    def write_times(self) -> list:
        """Compile available write times.

        :return: list of available write times
        :rtype: list(str)

        """
        pass

    @abstractmethod
    def field_names(self) -> dict:
        """Create a dictionary containing availale fields

        :return: dictionary with write times as keys and
            field names as values
        :rtype: dict(str:list(str))

        """
        pass

    @abstractmethod
    def load_snapshot(
        self,
        field_name: str,
        time: str,
        dtype: str,
        start_at: int,
        batch_size: int
    ) -> Tensor:
        """Load the snapshot of a single field.

        :param field_name: name of the field to load
        :type field_name: str
        :param time: snapshot time
        :type time: str
        :param dtype: type of torch tensor, e.g. `float32` or `float64`
        :type dtype: str
        :param start_at: index at which to start a batch
        :type start_at: int
        :param batch_size: batch size
        :type batch_size: int
        :return: field values
        :rtype: Tensor

        """
        pass

    @abstractmethod
    def get_vertices(self, start_at: int = 0, batch_size: int = BIG_INT) -> Tensor:
        """Get the vertices at which field values are defined.

        :return: coordinates of vertices
        :rtype: Tensor

        """
        pass

    @abstractmethod
    def get_weights(self, start_at: int = 0, batch_size: int = BIG_INT) -> Tensor:
        """Get the weights for field values.

        In a standard finite volume method, the weights are
        the cell volumes. For other methods, the definition
        of the weight is described in the Dataloader implementation.

        :return: weight for field values
        :rtype: Tensor
        
        """
        pass
