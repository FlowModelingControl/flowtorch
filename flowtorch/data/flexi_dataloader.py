r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FLEXIDataloader` class allows to load Flexi_ simulation data
from CGNS-HDF5 files.

.. _Flexi: https://www.flexi-project.org/
"""

from .dataloader import Dataloader
import torch as pt

BIG_INT = 1e15


class FLEXIDataloader(Dataloader):
    r"""

    """

    def __init__(self, path: str, dtype: str):
        self._path = path

    def write_times(self):
        pass

    def field_names(self):
        pass

    def load_snapshot(self, field_name: str, time: str, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        pass

    def get_vertices(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        pass

    def get_weights(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        pass
