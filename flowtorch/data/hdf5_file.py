r"""Module to read and write the internal flowTorch data format.

The :class:`HDF5Writer` class allows to write field and mesh data
into an HDF5 file. It also creates XDMF files for postprocessing
in ParaView. The :class:`HDF5Dataloader` is a concrete :class:`Dataloader`
class that allows efficient batch access to simulation data stored in the
internal flowTorch data format.
"""

from .dataloader import Dataloader
from .xdmf import XDMFWriter
from os.path import exists, isfile
from h5py import File
from mpi4py import MPI
import torch as pt


CONST_GROUP = "constant"
VAR_GROUP = "variable"
VERTICES_DS = "vertices"
CONNECTIVITY_DS = "connectivity"


class HDF5Dataloader(Dataloader):
    """
    """

    def __init__(self, file: str, dtype: str = pt.float32):
        """

        :param path: [description]
        :type path: [type]
        :param file_name: [description], defaults to FILE_NAME
        :type file_name: [type], optional
        :param dtype: [description], defaults to pt.float32
        :type dtype: [type], optional
        """
        pass

    def write_times(self):
        pass

    def field_names(self):
        pass

    def load_snapshot(self):
        pass

    def get_vertices(self):
        pass

    def get_weights(self):
        pass


class HDF5Writer(object):
    def __init__(self, file: str, dtype: str = pt.float32):
        self._file = File(file, "a", driver="mpio", comm=MPI.COMM_WORLD)
        self._dtype = dtype

    def write(self, field: pt.Tensor, name: str, time: str):
        ds_name = VAR_GROUP + "/{:s}/{:s}".format(time, name)
        self._file.create_dataset(ds_name, data=field.numpy(), dtype=self._dtype)

    def write_xdmf(self):
        pass

    def write_const(self, field: pt.Tensor, name: str):
        ds_name = CONST_GROUP + "/{:s}".format(name)
        self._file.create_dataset(ds_name, data=field.numpy(), dtype=self._dtype)


class FOAM2HDF5(object):
    def __init__(self, path):
        pass

    def convert(self, path):
        pass
