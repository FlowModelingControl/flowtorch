r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation solder. Currently, only the ESI branch
of OpenFOAM is supported (v1912, v2006).
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

SIZE_OF_CHAR = struct.calcsize("c")
SIZE_OF_DOUBLE = struct.calcsize("d")


class FOAMDataloader(Dataloader):
    r"""

    The structure of some methods is based on *ofpp* by Xu Xianghua
    (https://github.com/xu-xianghua/ofpp).

    """

    def __init__(self, path, dtype=pt.float32):
        r"""[summary]

        :param path: [description]
        :type path: [type]
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
        return self._case._time_folders

    def field_names(self):
        return self._case._field_names

    def load_snapshot(self, field_name, time, start_at=0, batch_size=1e15):
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

    def get_vertices(self):
        pass

    def get_weights(self):
        pass


class FOAMCase():
    """Class to access and parse OpenFOAM cases.
    """

    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            sys.exit("Error: could not find case {:s}".format(self._path))
        self._distributed = self._eval_distributed()
        self._processors = self._eval_processors()
        self._time_folders = self._eval_write_times()
        self._field_names = self._eval_field_names()

    @main_bcast
    def _eval_distributed(self):
        proc_dirs = glob.glob(self._path + "/processor*")
        return len(proc_dirs) > 0

    @main_bcast
    def _eval_processors(self):
        if self._distributed:
            return len(glob.glob(self._path + "/processor*"))
        else:
            return 1

    @main_bcast
    def _eval_write_times(self):
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
    def _eval_field_names(self):
        """
        assumption if distributed: all processor folders have the
        same fields per time step; determine field names from
        processor zero
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

    def build_file_path(self, field_name, time, processor):
        """
        """
        if self._distributed:
            file_path = (
                self._path +
                "/processor{:d}/{:s}/{:s}".format(processor, time, field_name)
            )
        else:
            file_path = self._path + "/{:s}/{:s}".format(time, field_name)
        return file_path
