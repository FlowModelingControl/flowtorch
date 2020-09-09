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
    def load_snapshot(
        self, field_name: str, time: str, dtype: str, start_at: int, batch_size: int
    ) -> Tensor:
        pass
