r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`MAIADataloader` class allows to load Maia simulation
data stored in NetCDF format.
"""
from .dataloader import Dataloader
from .mpi_tools import main_bcast
import glob
import torch as pt

BIG_INT = 1e15


class MAIADataloader(Dataloader):
    """

    :param Dataloader: [description]
    :type Dataloader: [type]
    """

    def __init__(self, path: str, dtype: str = pt.float32):
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
