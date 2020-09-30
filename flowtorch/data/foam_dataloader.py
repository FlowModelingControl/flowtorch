r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation folder. Currently, only the ESI-OpenCFD
branch of OpenFOAM is supported (v1912, v2006). The :class:`FoamCase`
class assembles information about the folder and file structure
of a simulation.
"""

from .dataloader import Dataloader
from .mpi_tools import main_bcast
import glob
import os
import struct
import sys
import torch as pt


FIELD_TYPE_DIMENSION = {
    b"volScalarField": 1,
    b"volVectorField": 3
}

MAX_LINE_HEADER = 20
MAX_LINE_INTERNAL_FIELD = 25
BIG_INT = 1e15

SIZE_OF_CHAR = struct.calcsize("c")
SIZE_OF_DOUBLE = struct.calcsize("d")


class FOAMDataloader(Dataloader):
    r"""Loads fields in native OpenFOAM format.

    The project ofpp_ by Xu Xianghua has been a great
    help to implement some of the methods.

    .. _ofpp: https://github.com/xu-xianghua/ofpp

    """

    def __init__(self, path: str, dtype: str=pt.float32):
        r"""Create a FOAMDataloader instance from a path.

        :param path: path to an OpenFOAM simulation folder.
        :type path: str
        :param dtype: tensor type; default is single precision `float32`
        :type dtype: str

        """
        self._case = FOAMCase(path)
        self._dtype = dtype

    def _parse_data(self, data):
        field_type = self._field_type(data[:MAX_LINE_HEADER])
        try:
            if self._is_binary(data[:MAX_LINE_HEADER]):
                field_data = self._unpack_internalfield_binary(
                    data, FIELD_TYPE_DIMENSION[field_type]
                )
            else:
                field_data = self._unpack_internalfield_ascii(
                    data, FIELD_TYPE_DIMENSION[field_type]
                )
        except Exception as e:
            print(e)
        else:
            return field_data

    def _is_binary(self, data):
        for line in data:
            if b"format" in line:
                if b"binary" in line:
                    return True
                else:
                    return False
        return False

    def _find_nonuniform(self, data):
        for i, line in enumerate(data):
            if b"nonuniform" in line:
                return i, int(data[i+1])
        return 0, 0

    def _field_type(self, data):
        for line in data:
            if b"class" in line:
                for field_type in FIELD_TYPE_DIMENSION.keys():
                    if field_type in line:
                        return field_type
                return None
        return None

    def _unpack_internalfield_ascii(self, data, dim):
        start, n_values = self._find_nonuniform(data[:MAX_LINE_INTERNAL_FIELD])
        start += 3
        if dim == 1:
            return pt.tensor([float(line) for line in data[start:start + n_values]], dtype=self._dtype)
        else:
            return pt.tensor(
                [list(map(float, line[1:-2].split()))
                 for line in data[start:start + n_values]],
                dtype=self._dtype
            )

    def _unpack_internalfield_binary(self, data, dim):
        start, n_values = self._find_nonuniform(data[:MAX_LINE_INTERNAL_FIELD])
        start += 2
        buffer = b"".join(data[start:])
        values = struct.unpack(
            "{}d".format(dim*n_values),
            buffer[SIZE_OF_CHAR:SIZE_OF_CHAR+SIZE_OF_DOUBLE*n_values*dim]
        )
        if dim > 1:
            return pt.tensor(values, dtype=self._dtype).reshape(n_values, dim)
        else:
            return pt.tensor(values, dtype=self._dtype)

    def write_times(self):
        """Returns the output of :func:`FOAMCase._eval_write_times`
        """
        return self._case._time_folders

    def field_names(self):
        """Returns the output of :func:`FOAMCase._eval_field_names`
        """
        return self._case._field_names

    def load_snapshot(self, field_name, time, start_at=0, batch_size=BIG_INT) -> pt.Tensor:
        file_paths = []
        if self._case._distributed:
            for proc in range(self._case._processors):
                file_paths.append(
                    self._case.build_file_path(field_name, time, proc))
        else:
            file_paths.append(self._case.build_file_path(field_name, time, 0))
        field_data = []
        for file_path in file_paths:
            try:
                with open(file_path, "rb") as file:
                    field_data.append(self._parse_data(file.readlines()))
            except Exception as e:
                print("Error: could not read file {:s}".format(file_path))
                print(e)
        joint_data = pt.cat(field_data)
        return joint_data[start_at:min(batch_size, joint_data.size()[0])]

    def get_vertices(self) -> pt.Tensor:
        """Get vertices at which field values are defined.

        In OpenFOAM, all field are defined at the control volume's
        center. Therefore, get vertices returns the cell center locations.

        :returns: control volume centers
        :rtype: Tensor

        """
        pass

    def get_weights(self):
        pass


class FOAMCase(object):
    """Class to access and parse OpenFOAM cases.

    Most of the attributes and methods are private because they are
    typically accessed via a :class:`FOAMDataloader` instance.

    .. automethod:: _eval_distributed
    .. automethod:: _eval_processors
    .. automethod:: _eval_write_times
    .. automethod:: _eval_field_names
    """

    def __init__(self, path):
        """Create a `FOAMCase` instance based on a path.

        :param path: path to OpenFOAM simulation case
        :type path: str

        """
        self._path = path
        if not os.path.exists(self._path):
            sys.exit("Error: could not find case {:s}".format(self._path))
        if self._path[-1] == "/":
            self._path = self._path[:-1]
        self._distributed = self._eval_distributed()
        self._processors = self._eval_processors()
        self._time_folders = self._eval_write_times()
        self._field_names = self._eval_field_names()

    @main_bcast
    def _eval_distributed(self) -> bool:
        """Check if the simulation case is distributed (parallel).

        .. warning::
            Collated output is currently not supported/not detected.

        :return: `True` if distributed
        :rtype: bool

        """
        proc_dirs = glob.glob(self._path + "/processor*")
        return len(proc_dirs) > 0

    @main_bcast
    def _eval_processors(self) -> int:
        """Get number of processor folders.

        :return: number of processor folders or 1 for serial runs
        :rtype: int

        """
        if self._distributed:
            return len(glob.glob(self._path + "/processor*"))
        else:
            return 1

    @main_bcast
    def _eval_write_times(self) -> list:
        """Assemble a list of all write times.

        :return: a list of all time folders
        :rtype: list(str)

        .. warning::
            For distributed simulations, it is assumed that all processor
            folders contain the same time folders.
        """
        if self._distributed:
            time_path = self._path + "/processor0"
        else:
            time_path = self._path
        dirs = [folder for folder in os.listdir(time_path) if
                os.path.isdir(os.path.join(time_path, folder))]
        time_dirs = []
        for entry in dirs:
            try:
                _ = float(entry)
                time_dirs.append(entry)
            except:
                pass
        if len(time_dirs) < 2:
            print(
                "Warning: found only one or less time folders in {:s}"
                .format(self._path)
            )
        return sorted(time_dirs, key=float)

    @main_bcast
    def _eval_field_names(self) -> dict:
        """Get a dictionary of all fields and files in all time folders.

        .. warning::
            For distributed cases, only *processor0* is evaluated. The fields
            for all other processors are assumed to be the same.

        :return: dictionary with write times as keys and the field names
            for each time as values
        :rtype: dict

        """
        all_time_folders = [
            self.build_file_path("", time, 0)
            for time in self._time_folders
        ]
        all_fields = {}
        for i, folder in enumerate(all_time_folders):
            all_fields[self._time_folders[i]] = [
                field for field in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, field))
            ]
        return all_fields

    def build_file_path(self, field_name, time, processor=0) -> str:
        """Create the path to file inside the time folder of a simulation.

        :param field_name: name of the field or file, e.g., \"U\" or \"p\"
        :type field_name: str
        :param time: name of the time folder, e.g., \"0.01\"
        :type time: str
        :param processor: processor folder to load the data from; ignored
            in serial simulation cases; defaults to `0`
        :type processor: int, optional
        :return: path to file inside a time folder
        :rtype: str

        Examples

        >>> from flowtorch.data import FOAMCase
        >>> case = FOAMCase("./cavity_binary_parallel/")
        >>> case._distributed
        True
        >>> case._processors
        4
        >>> case._time_folders
        ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
        >>> case._field_names
        {'0': ['U', 'p'], '0.1': ['U', 'p', 'phi'], '0.2': ['U', 'p', 'phi'], '0.3': ['U', 'p', 'phi'], '0.4': ['U', 'p', 'phi'], '0.5': ['U', 'p', 'phi']}
        >>> case.build_file_path("U", "0.1", 1)
        './cavity_binary_parallel/processor1/0.1/U'

        """
        if self._distributed:
            file_path = (
                self._path +
                "/processor{:d}/{:s}/{:s}".format(processor, time, field_name)
            )
        else:
            file_path = self._path + "/{:s}/{:s}".format(time, field_name)
        return file_path
