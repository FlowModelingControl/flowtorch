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
import torch as pt

FILE_NAME = "flowtorch.hdf5"

class HDF5Dataloader(Dataloader):
    """
    """
    def __init__(self, path, file_name=FILE_NAME, dtype=pt.float32):
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
    def __init__(self, path, file_name=FILE_NAME, dtype=pt.float32):
        pass

    def write(self, field_name, data):
        pass

    def write_xdmf(self):
        pass

    def write_mesh(self):
        pass


class FOAM2HDF5(object):
    def __init__(self, path):
        pass

    def convert(self, path):
        pass
