r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation folder. Currently, only the ESI-OpenCFD
branch of OpenFOAM is supported (v1912, v2006). The :class:`FOAMCase`
class assembles information about the folder and file structure
of a simulation. The :class:`FOAMMesh` allows to loads and parses
the finite volume mesh.
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

POLYMESH_PATH = "constant/polyMesh/"
MESH_FILES = ["points", "owner", "neighbour", "faces", "boundary"]


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

    def __init__(self, path: str, dtype: str = pt.float32):
        r"""Create a FOAMDataloader instance from a path.

        :param path: path to an OpenFOAM simulation folder.
        :type path: str
        :param dtype: tensor type; default is single precision `float32`
        :type dtype: str

        """
        self._case = FOAMCase(path)
        self._mesh = FOAMMesh(self._case)
        self._dtype = dtype

    def _parse_data(self, data):
        field_type = self._field_type(data[:MAX_LINE_HEADER])
        try:
            if self._case._is_binary(data[:MAX_LINE_HEADER]):
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

    def load_snapshot(self, field_name: str, time: str, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
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

    def get_vertices(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        """Get vertices at which field values are defined.

        In OpenFOAM, all field are defined at the control volume's
        center. Therefore, get vertices returns the cell center locations.

        :returns: control volume centers
        :rtype: Tensor

        """
        centers = self._mesh.get_cell_centers()
        return centers[start_at:min(batch_size, centers.size()[0])]

    def get_weights(self, start_at: int = 0, batch_size: int = BIG_INT) -> pt.Tensor:
        """Get cell volumes.

        For results obtained using a finite volume method with co-located
        arrangement, a sensible weight for a cell-centered value is the cell
        volume.

        :return: cell volumes
        :rtype: pt.Tensor
        """
        volumes = self._mesh.get_cell_volumes()
        return volumes[start_at:min(batch_size, volumes.size()[0])]


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
        if not self._check_mesh_files():
            sys.exit("Error: could not find valid mesh in case {:s}".format(
                self._case._path))

    def _is_binary(self, header):
        for line in header:
            if b"format" in line:
                if b"binary" in line:
                    return True
                else:
                    return False
        return False

    @main_bcast
    def _check_mesh_files(self):
        """Check if all mesh files are available.
        """
        if self._distributed:
            files_found = []
            for proc in range(self._processors):
                files_found += [
                    os.path.isfile(
                        self._path + "/processor{:d}/".format(proc)
                        + POLYMESH_PATH + mesh_file
                    )
                    for mesh_file in MESH_FILES
                ]
        else:
            files_found = [
                os.path.isfile(
                    self._path + "/" + POLYMESH_PATH + mesh_file
                )
                for mesh_file in MESH_FILES
            ]
        return all(files_found)

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

    def build_file_path(self, field_name: str, time: str, processor: int = 0) -> str:
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


class FOAMMesh(object):
    """Class to load load and process OpenFOAM meshes.

    OpenFOAM stores the finite volume mesh as a collection of several
    files located in *constant/polyMesh* or in *processorXX/constant/polyMesh*
    for serial and distributed cases, respectively. Even though OpenFOAM
    is a cell-centered finite volume method, the cell-centers and volumes are
    not explicitly stores. Instead, a so-called face-addressing storage is used.
    All internal faces have an owner cell and a neighbor cell. Boundary faces only
    have an owner cell. The mesh attributes are defined in several files:

    - **points**: list of vertices forming cell faces; the list index of a point is used as label
    - **faces**: list of all cell faces defined as point labels
    - **owner**: list of cell labels that are face owners
    - **neighbour**: list of cell label that are face neighbors; BE spelling
    - **boundary**: definition of faces belonging to a patch

    .. warning::
            Dynamically changing meshes are currently not supported.

    .. automethod:: _compute_cell_centers_and_volumes

    """

    def __init__(self, case: FOAMCase, dtype: str = pt.float32):
        self._case = case
        self._dtype = dtype
        self._cell_centers = None
        self._cell_volumes = None

    def _get_list_length(self, data):
        """Find list length of points, faces, and cells.

        :param data: number of elements in OpenFOAM list and line
            with first list entry
        :type data: tuple(int, int)
        """
        for i, line in enumerate(data):
            try:
                n_entries = int(line)
            except:
                pass
            else:
                return i, n_entries
        return 0, 0

    def _parse_points(self, mesh_path):
        """Parse mesh vertices defined in *constant/polyMesh/points*.
        """
        with open(mesh_path + "points", "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            if self._case._is_binary(data[:MAX_LINE_HEADER]):
                return pt.Tensor([0, 0, 0]).unsqueeze(-1)
            else:
                start += 2
                return pt.tensor(
                    [list(map(float, line[1:-2].split()))
                     for line in data[start:start + 4]],
                    dtype=self._dtype
                )

    def _parse_faces(self, mesh_path):
        """Parse cell faces stored in in *constant/polyMesh/faces*.
        """
        pass

    def _parse_owners_and_neighbors(self, mesh_path):
        """Parse face owners and neighbors.

        - owners are parsed from *constant/polyMesh/owner*
        - neighbors are parsed from *constant/polyMesh/neighbour*

        """
        pass

    def _compute_cell_centers_and_volumes(self, mesh_path):
        """Compute the cell centers and volumes of an OpenFOAM mesh.

        The implemented algorithm is the same as in makeCellCentresAndVols_.
        OpenFOAM uses 
        The following steps are involved:

        1. compute an estimate of the cell center as the sum over all face centers of a cell
        2. 

        .. _makeCellCentresAndVols: https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1primitiveMesh.html

        """
        points = self._parse_points(mesh_path)
        n_points_faces, faces = self._parse_faces(mesh_path)
        owners, neighbors = self._parse_owners_and_neighbors(mesh_path)

    def _load_mesh(self) -> pt.Tensor:
        """[summary]
        """
        if self._case._distributed:
            proc_data = []
            for proc in range(self._case._processors):
                mesh_location = self._case._path + "/" + POLYMESH_PATH
                proc_data.append(
                    self._compute_cell_centers_and_volumes(mesh_location))
            centers = pt.cat(list(zip(*proc_data))[0])
            volumes = pt.cat(list(zip(*proc_data))[1])
        else:
            mesh_location = self._case._path + "/" + POLYMESH_PATH
            centers, volumes = self._compute_cell_centers_and_volumes(
                mesh_location)
        self._cell_centers = centers
        self._cell_volumes = volumes

    def get_cell_centers(self) -> pt.Tensor:
        """Return or compute and return control volume centers.

        :return: control volume centers
        :rtype: pt.Tensor
        """
        if self._cell_centers == None:
            self._load_mesh()
        return self._cell_centers

    def get_cell_volumes(self) -> pt.Tensor:
        """Return or compute and return cell volumes.

        :return: cell volumes
        :rtype: pt.Tensor
        """
        if self._cell_volumes == None:
            self._load_mesh()
        return self._cell_volumes
