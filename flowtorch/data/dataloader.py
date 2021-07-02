"""Definition of a common interface for all dataloaders.

This abstract base class should be used as parent class when
defining new dataloaders, e.g., to support additional file
formats.
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict
from torch import Tensor


class Dataloader(ABC):
    r"""Abstract base class to define a common interface for dataloaders.
    """

    @abstractmethod
    def load_snapshot(self, field_name: str, time: str) -> Tensor:
        """Load the snapshot of a single field.

        :param field_name: name of the field to load
        :type field_name: str
        :param time: snapshot time
        :type time: str
        :return: field values
        :rtype: Tensor

        """
        pass

    @abstractproperty
    def write_times(self) -> List[str]:
        """Available write times.

        :return: list of available write times
        :rtype: List[str]

        """
        pass

    @abstractproperty
    def field_names(self) -> Dict[str, List[str]]:
        """Create a dictionary containing availale fields

        :return: dictionary with write times as keys and
            field names as values
        :rtype: Dict[str, List[str]]

        """
        pass

    @abstractproperty
    def vertices(self) -> Tensor:
        """Get the vertices at which field values are defined.

        :return: coordinates of vertices
        :rtype: Tensor

        """
        pass

    @abstractproperty
    def weights(self) -> Tensor:
        """Get the weights for field values.

        In a standard finite volume method, the weights are
        the cell volumes. For other methods, the definition
        of the weight is described in the Dataloader implementation.

        :return: weight for field values
        :rtype: Tensor

        """
        pass
