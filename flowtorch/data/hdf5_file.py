r"""Module to read and write the internal flowTorch data format.

The :class:`HDF5Writer` class allows to write field and mesh data
into an HDF5 file. It also creates XDMF files for postprocessing
in ParaView. The :class:`HDF5Dataloader` is a concrete :class:`Dataloader`
class that allows efficient batch access to simulation data stored in the
internal flowTorch data format.
"""

from .dataloader import Dataloader
from .foam_dataloader import FOAMCase, FOAMMesh, FOAMDataloader
from .xdmf import XDMFWriter
from os.path import exists, isfile
from h5py import File
from mpi4py import MPI
import torch as pt


CONST_GROUP = "constant"
VAR_GROUP = "variable"
VERTICES_DS = "vertices"
CONNECTIVITY_DS = "connectivity"

dtype_conversion = {
    pt.float32: "f4",
    pt.float64: "f8",
    pt.int32: "i4",
    pt.int64: "i8",
    "float32": "f4",
    "float64": "f8",
    "int32": "i4",
    "int64": "i8"
}


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
    """Class to write flowTorch data to HDF5 file.

    Two types of data are supported:
    - variable: (field) data that changes with times, e.g, snapshots
    - constant: constant data like mesh vertices or cell volumes

    A XDMF accessor file can be created to support visual post-processing
    with ParaView and other XDMF-compatible software packages.
    """

    def __init__(self, file: str):
        """Construct :class:`HDF5Writer` object based on file name.

        :param file: path and file name to HDF5 file.
        :type file: str
        """
        self._file_path = file
        self._file = File(file, mode="a", driver="mpio", comm=MPI.COMM_WORLD)

    def __del__(self):
        """Destructor to ensure that HDF5 file is closed.
        """
        self._file.close()

    def write(self, field: pt.Tensor, name: str, time: str, dtype: str = pt.float32):
        """Write variable (time-dependent) field data to HDF5 file.

        :param field: field data
        :type field: pt.Tensor
        :param name: field name in dataset
        :type name: str
        :param time: sample time; becomes HDF5 group
        :type time: str
        :param dtype: datatype, defaults to pt.float32
        :type dtype: str, optional

        .. warning::
            If a dataset with the same name already exists in the HDF5 file,
            the existing dataset is deleted.
        """
        ds_name = VAR_GROUP + "/{:s}/{:s}".format(time, name)
        if dtype in dtype_conversion.keys():
            if ds_name in self._file:
                del self._file[ds_name]
            self._file.create_dataset(
                ds_name,
                data=field.numpy(),
                dtype=dtype_conversion[dtype]
            )
        else:
            print(
                "Warning: invalid data type {:s} for field {:s}. Skipping field.".format(
                    str(dtype), name)
            )

    def write_xdmf(self):
        pass

    def write_const(self, field: pt.Tensor, name: str, dtype: str = pt.float32):
        """Write constant data to HDF5 file.

        :param field: field data
        :type field: pt.Tensor
        :param name: field name in dataset
        :type name: str
        :param dtype: datatype, defaults to pt.float32
        :type dtype: str, optional

        .. warning::
            If a dataset with the same name already exists in the HDF5 file,
            the existing dataset is deleted.
        """
        ds_name = CONST_GROUP + "/{:s}".format(name)
        if dtype in dtype_conversion.keys():
            if ds_name in self._file:
                del self._file[ds_name]
            self._file.create_dataset(
                ds_name,
                data=field.numpy(),
                dtype=dtype_conversion[dtype]
            )
        else:
            print(
                "Warning: invalid data type {:s} for field {:s}. Skipping field.".format(
                    str(dtype), name)
            )


class FOAM2HDF5(object):
    def __init__(self, path):
        self._case = FOAMCase(path)
        self._loader = FOAMDataloader(path)

    def convert(self, path):
        pass
