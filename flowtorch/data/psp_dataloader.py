r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`PSPDataloader` class allows to load
instationary pressure-sensitive paint (iPSP_) data provided by
DLR (Deutsches Luft- und Raumfahrtzentrum).

.. _iPSP: https://www.dlr.de/as/en/desktopdefault.aspx/tabid-183/251_read-13334/
"""

from .dataloader import Dataloader
from .mpi_tools import main_bcast
from os.path import exists
from h5py import File
from mpi4py import MPI
from math import ceil
import torch as pt
import sys

BIG_INT = 1e15
IGNORE_KEYS = ["Info"]
DEFAULT_FIELD_NAME = "cp"
DEFAULT_DATA_NAME = "Images"


class PSPDataloader(Dataloader):
    r"""

    """

    def __init__(self, path: str, dtype: str = pt.float32):
        self._path = path
        self._dtype = dtype
        if exists(self._path):
            self._file = File(self._path, mode="r",
                              driver="mpio", comm=MPI.COMM_WORLD)
        else:
            sys.exit("Error: could not find file {:s}".format(self._path))
        self._dataset_names = self._get_dataset_names()
        self._dataset = self._dataset_names[0]
        self._attributes = self._get_attributes()
        self._write_times = self._get_write_times()

    @main_bcast
    def _get_dataset_names(self) -> list:
        all_keys = self._file.keys()
        return [key for key in all_keys if not key in IGNORE_KEYS]

    @main_bcast
    def _get_attributes(self) -> dict:
        attributes = {
            "Frequency": 1000.0
        }
        return attributes

    @main_bcast
    def _get_write_times(self):
        freq = self._attributes["Frequency"]
        data_path = "/".join([self._dataset, DEFAULT_DATA_NAME])
        n_snapshot = self._file[data_path].shape[-1]
        return ["{:2.6f}".format(float(i)/freq) for i in range(n_snapshot)]

    def _time_to_index(self, time):
        freq = self._attributes["Frequency"]
        return int(ceil(float(time) * freq))

    def select_dataset(self, name: str):
        if name in self._dataset_names:
            self._dataset = name
        else:
            possible_names = "\n".join(self._dataset_names)
            print(
                """Warning cannot select dataset {:s}. Available datasets are\n{:s}"""
                .format(name, possible_names)
            )
        print("Selected dataset: {:s}".format(self._dataset))

    def write_times(self):
        return self._write_times

    def field_names(self):
        field_list = [DEFAULT_FIELD_NAME]
        return dict(
            zip(self._write_times, [field_list]*len(self._write_times))
        )

    def load_snapshot(self, field_name: str, time: str, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        data_path = "/".join([self._dataset, DEFAULT_DATA_NAME])
        if data_path in self._file:
            field = pt.tensor(
                self._file[data_path][:, :, self._time_to_index(time)],
                dtype=self._dtype
            ).flatten()
        else:
            sys.exit("Dataset {:s} not found.".format(data_path))
        return field[start_at:min(batch_size, field.size()[0])]

    def get_vertices(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        data_path = "/".join([self._dataset, "CoordinatesX"])
        shape = self._file[data_path].shape
        n_vertices = shape[0] * shape[1]
        vertices = pt.zeros((n_vertices, 3), dtype=self._dtype)
        for i, axis in enumerate(["CoordinatesX", "CoordinatesY", "CoordinatesZ"]):
            data_path = "/".join([self._dataset, "CoordinatesX"])
            vertices[:, i] = pt.tensor(
                self._file[data_path][:, :], dtype=self._dtype).flatten()
        return vertices[start_at:min(batch_size, vertices.size()[0]), :]

    def get_weights(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        data_path = "/".join([self._dataset, "CoordinatesX"])
        shape = self._file[data_path].shape
        return pt.ones(shape[0]*shape[1], dtype=self._dtype)
