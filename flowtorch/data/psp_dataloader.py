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
from typing import List
import torch as pt
import numpy as np
import sys


COORDINATE_KEYS = ["CoordinatesX", "CoordinatesY", "CoordinatesZ"]
WEIGHT_KEY = "Mask"
IGNORE_KEYS = ["Info", "Parameter", "ParameterDescription", "TimeValues", WEIGHT_KEY] + COORDINATE_KEYS


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
        self._dataset_names = self.get_dataset_names()
        if len(self._dataset_names) > 0:
            self._dataset = self.get_dataset_names()[0]
        else:
            print(
                "Warning: could not find dataset in file {:s}".format(self._file))
            self._dataset = None
        self._attributes = self._get_attributes()
        self._write_times = self._get_write_times()

    

    @main_bcast
    def _get_attributes(self) -> dict:
        # TODO: read frequency from file once format is finalized
        attributes = {
            "Frequency": 2000.0
        }
        return attributes

    @main_bcast
    def _get_write_times(self):
        freq = self._attributes["Frequency"]
        fields = self.field_names()
        data_path = "/".join([self._dataset, fields[0]])
        n_snapshot = self._file[data_path].shape[-1]
        return ["{:2.6f}".format(float(i)/freq) for i in range(n_snapshot)]

    def _time_to_index(self, time):
        freq = self._attributes["Frequency"]
        return int(round(float(time) * freq, 0))

    @main_bcast
    def get_dataset_names(self) -> list:
        all_keys = self._file.keys()
        return [key for key in all_keys if not key in IGNORE_KEYS]

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
        set_keys = self._file[self._dataset].keys()
        return [key for key in set_keys if key not in IGNORE_KEYS]

    def load_snapshot(self, field_name: str, time: List[str]) -> pt.Tensor:
        data_path = "/".join([self._dataset, field_name])
        indices = np.array([self._time_to_index(t) for t in time])
        if data_path in self._file:
            field = pt.tensor(
                self._file[data_path][:, :, indices],
                dtype=self._dtype
            )
        else:
            sys.exit("Dataset {:s} not found.".format(data_path))
        return field

    def get_vertices(self) -> pt.Tensor:
        data_path = "/".join([self._dataset, "CoordinatesX"])
        shape = self._file[data_path].shape
        vertices = pt.zeros((*shape, 3), dtype=self._dtype)
        for i, axis in enumerate(COORDINATE_KEYS):
            data_path = "/".join([self._dataset, axis])
            vertices[:, :, i] = pt.tensor(
                self._file[data_path][:, :], dtype=self._dtype)
        return vertices

    def get_weights(self) -> pt.Tensor:
        data_path = "/".join([self._dataset, "Mask/"])
        weights = pt.tensor(self._file[data_path][:], dtype=self._dtype)
        return weights
