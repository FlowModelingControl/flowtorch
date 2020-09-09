r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation solder. Currently, only the ESI branch
of OpenFOAM is supported (v1912, v2006).
"""

from .dataloader import Dataloader
from .mpi_tools import main_bcast
import glob
import os
import sys
import torch as pt
import numpy as np


class FOAMDataloader(Dataloader):
    r"""

    """

    def __init__(self, path):
        r"""[summary]

        :param path: [description]
        :type path: [type]
        """
        self._case = FOAMCase(path)

    def _parse_data(self, data, mode):
        _, line = self._find_line_by_keyword(data, b"class")
        field_type = line.split(b" ")[-1][:-1]
        field_parsers = {
            b"volScalarField": self._parse_volScalarField,
            b"volVectorField": self._parse_volVectorField
        }
        try:
            field_data = field_parsers[field_type](data, mode)
        except:
            print(
                "Error: field type {:s} is not supported"
                .format(field_type)
            )
        else:
            return field_data

    def _find_line_by_keyword(self, data, keyword):
        found = False
        line_i = -1
        while not found and line_i < len(data):
            line_i += 1
            if keyword in data[line_i]:
                found = True
        return line_i, data[line_i]

    def _parse_volScalarField(self, data, mode):
        pass

    def _parse_volVectorField(self, data, mode):
        print("Parsing vector field...")
        line_i, line = self._find_line_by_keyword(data, b"internalField")
        n_values = int(data[line_i + 1])
        print("{:d} internal values".format(n_values))
        line_j, _ = self._find_line_by_keyword(data, b"boundaryField")
        print(len(data[line_i + 2:]))
        print(len(data[line_i + 2:line_j - 1]))
        print(data[line_i + 2])
        print(data[line_i + 3])

    def write_times(self):
        return self._case.write_times()

    def field_names(self):
        return self._case.field_names()

    def load_snapshot(self, field_name, time, start_at, batch_size):
        file_paths = []
        if self._case.distributed():
            for proc in range(self._case.processors()):
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
        return pt.cat(field_data)

    def load_mesh(self):
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
        if self.distributed():
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


    def distributed(self):
        return self._distributed

    def processors(self):
        return self._processors

    def write_times(self):
        """Extract the write times from the folder names.
        """
        return self._time_folders

    def field_names(self):
        """
        """
        return self._field_names

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
