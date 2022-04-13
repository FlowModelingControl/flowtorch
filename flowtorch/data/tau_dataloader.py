r"""Direct access to TAU simulation data.

The DRL (Deutsches Luft- und Raumfahrtzentrum) TAU_ code saves
snapshots in the NetCFD format. The :class:`TAUDataloader` is a
wrapper around the NetCFD Python bindings to simplify the access
to snapshot data.

.. _TAU: https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/

"""
# standard library packages
from glob import glob
from typing import List, Dict, Tuple, Union, Set
# third party packages
from netCDF4 import Dataset
import torch as pt
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_list_or_str, check_and_standardize_path

PMESH_NAME = "dualgrid_domain_{:s}_grid_1"
PVERTEX_KEY = "pcoord"
PWEIGHT_KEY = "pvolume"
PADD_POINTS_KEY = "addpoint_idx"
PLOCAL_ID_KEY = "local_id"
VERTEX_KEYS = ("points_xc", "points_yc", "points_zc")
WEIGHT_KEY = "volume"


class TAUDataloader(Dataloader):
    """Load TAU simulation data.

    TAU simulations output results in several netCDF files, one for each write
    time. The mesh is stored in a separated file with the extension *.grd*.
    Currently, the loader only enables access to field data but not to boundaries.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import TAUDataloader
    >>> path = DATASETS["tau_backward_facing_step"]
    >>> loader = TAUDataloader(path, mesh_path=path, base_name="sol.pval.unsteady_")
    >>> times = loader.write_times
    >>> fields = loader.field_names[times[0]]
    >>> fields
    ['density', 'x_velocity', 'y_velocity', ...]
    >>> density = loader.load_snapshot("density", times)
    >>> density.shape
    torch.Size([1119348, 10])

    """

    def __init__(self, solution_path: str, mesh_path: str, base_name: str,
                 distributed: bool = False, subfolders: bool = False,
                 dtype: str = DEFAULT_DTYPE):
        """Create loader instance from TAU simulation folder.

        :param solution_path: path to TAU simulation solution files
        :type solution_path: str
        :param mesh_path: path to TAU simulation mesh files
        :type mesh_path: str
        :param base_name: part of the solution file name before iteration count,
            e.g., base_name_ if the solution file is called base_name_i=0102_t=1.0
        :type base_name: str
        :param distributed: True if mesh and solution are distributed in domain
            files; defaults to False
        :type distributed: bool, optional
        :param subfolders: True if solution files are stored in dedicated folders
            for each write time; defaults to False
        type subfolders: bool, optional
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._sol_path = check_and_standardize_path(solution_path)
        self._mesh_path = check_and_standardize_path(mesh_path)
        self._base_name = base_name
        self._distributed = distributed
        self._subfolders = subfolders
        self._dtype = dtype
        if self._distributed:
            self._domain_ids = self._get_domain_ids()
        self._time_iter = self._decompose_file_name()
        self._mesh_data = self._load_mesh_data()

    def _get_domain_ids(self) -> List[str]:
        """Determine ids of processes domains.

        Grid and solution files of distributed simulations contain a string
        of the form *domain_id* in the filename. This function extracts the
        *id* from the grid files.

        :return: sorted list of available domain ids
        :rtype: List[str]
        """
        files = glob(f"{self._mesh_path}/{PMESH_NAME.format('*')}")
        ids = [s.split("grid_domain_")[-1].split("_")[0] for s in files]
        return sorted(ids, key=int)

    def _find_grid_file(self) -> str:
        """Determine the grid file's name in serial simulations.

        :raises FileNotFoundError: if no grid file is found
        :raises FileNotFoundError: if multiple grid files are found
        :return: name of the grid file
        :rtype: str
        """
        files = glob(f"{self._mesh_path}/*.grd")
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find mesh file (.grd) in {self._mesh_path}/")
        if len(files) > 1:
            raise FileNotFoundError(
                f"Found multiple mesh files (.grd) in {self._mesh_path}/")
        return files[0].split("/")[-1]

    def _decompose_file_name(self) -> Dict[str, str]:
        """Extract write time and iteration from file name.

        :raises FileNotFoundError: if no solution files are found
        :return: dictionary with write times as keys and the corresponding
            iterations as values
        :rtype: Dict[str, str]
        """
        if self._distributed:
            d0 = f".domain_{self._domain_ids[0]}"
            files = glob(
                f"{self._sol_path}/**/{self._base_name}i=*t=*{d0}", recursive=True)
        else:
            files = glob(
                f"{self._sol_path}/**/{self._base_name}i=*t=*", recursive=True)
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find solution files in {self._sol_path}/")
        time_iter = {}
        split_at = f".domain" if self._distributed else " "
        for f in files:
            t = f.split("t=")[-1].split(split_at)[0]
            i = f.split("i=")[-1].split("_t=")[0]
            time_iter[t] = i
        return time_iter

    def _file_name(self, time: str, suffix: str = "") -> str:
        """Create solution file name from write time.

        :param time: snapshot write time
        :type time: str
        :param suffix: suffix to append to the file name; used for decomposed
            simulations
        :type suffix: str, optional
        :return: name of solution file
        :rtype: str
        """
        itr = self._time_iter[time]
        if self._subfolders:
            path = f"{self._sol_path}/i={itr}_t={time}"
        else:
            path = f"{self._sol_path}"
        return f"{path}/{self._base_name}i={itr}_t={time}{suffix}"

    def _load_domain_mesh_data(self, pid: str) -> pt.Tensor:
        """Load vertices and volumes for a single processor domain.

        :param pid: domain id
        :type pid: str
        :return: tensor of size n_points x 4, where n_points is the number
            of unique cells in the domain, and the 4 columns contain the
            coordinates of the vertices (x, y, z) and the cell volumes
        :rtype: pt.Tensor
        """
        path = f"{self._mesh_path}/{PMESH_NAME.format(pid)}"
        with Dataset(path) as data:
            vertices = pt.tensor(data[PVERTEX_KEY][:], dtype=self._dtype)
            volumes = pt.tensor(data[PWEIGHT_KEY][:], dtype=self._dtype)
            local_ids = pt.tensor(data[PLOCAL_ID_KEY][:], dtype=pt.int64)
            n_add_points = data[PADD_POINTS_KEY].shape[0]

        n_points = volumes.shape[0] - n_add_points
        mask = pt.ones_like(volumes, dtype=pt.bool)
        mask[local_ids >= n_points] = False
        data = pt.zeros((n_points, 4), dtype=self._dtype)
        data[:, 0] = pt.masked_select(vertices[:, 0], mask)
        data[:, 1] = pt.masked_select(vertices[:, 1], mask)
        data[:, 2] = pt.masked_select(vertices[:, 2], mask)
        data[:, 3] = pt.masked_select(volumes, mask)
        return data

    def _load_mesh_data(self) -> pt.Tensor:
        """Load mesh vertices and cell volumes.

        :return: Tensor of dimension n_points x 4; the first three columns
            correspond to the x/y/z coordinates, and the 4th column contains
            the volumes
        :rtype: pt.Tensor
        """
        if self._distributed:
            return pt.cat(
                [self._load_domain_mesh_data(pid) for pid in self._domain_ids],
                dim=0
            )
        else:
            path = f"{self._mesh_path}/{self._find_grid_file()}"
            with Dataset(path) as data:
                vertices = pt.stack(
                    [pt.tensor(data[key][:], dtype=self._dtype)
                     for key in VERTEX_KEYS],
                    dim=-1
                )
                if WEIGHT_KEY in data.variables.keys():
                    weights = pt.tensor(
                        data.variables[WEIGHT_KEY][:], dtype=self._dtype)
                else:
                    print(
                        f"Warning: could not find cell volumes in file {path}")
                    weights = pt.ones(vertices.shape[0], dtype=self._dtype)
            return pt.cat((vertices, weights.unsqueeze(-1)), dim=-1)

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load a single snapshot of a single field from the netCDF4 file(s).

        :param field_name: name of the field
        :type field_name: str
        :param time: snapshot write time
        :type time: str
        :return: tensor holding the field values
        :rtype: pt.Tensor
        """
        if self._distributed:
            field = []
            for pid in self._domain_ids:
                path = self._file_name(time, f".domain_{pid}")
                with Dataset(path) as data:
                    field.append(
                        pt.tensor(
                            data.variables[field_name][:], dtype=self._dtype)
                    )
            return pt.cat(field, dim=0)
        else:
            path = self._file_name(time)
            with Dataset(path) as data:
                field = pt.tensor(
                    data.variables[field_name][:], dtype=self._dtype)
        return field

    def load_snapshot(self, field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")

        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [
                    pt.stack([self._load_single_snapshot(field, t)
                              for t in time], dim=-1)
                    for field in field_name
                ]
            else:
                return [
                    self._load_single_snapshot(field, time) for field in field_name
                ]
        # load single field
        else:
            if isinstance(time, list):
                return pt.stack(
                    [self._load_single_snapshot(field_name, t) for t in time],
                    dim=-1
                )
            else:
                return self._load_single_snapshot(field_name, time)

    @property
    def write_times(self) -> List[str]:
        return sorted(list(self._time_iter.keys()), key=float)

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """Find available fields in solution files.

        Available fields are determined by matching the number of
        weights with the length of datasets in the available
        solution files; for distributed cases, the fields are only
        determined based on *domain_0*.

        :return: dictionary with time as key and list of
            available solution fields as value
        :rtype: Dict[str, List[str]]
        """
        self._field_names = {}
        if self._distributed:
            n_points = self._load_domain_mesh_data(
                self._domain_ids[0]).shape[0]
            suffix = f".domain_{self._domain_ids[0]}"
        else:
            n_points = self.vertices.shape[0]
            suffix = ""
        for time in self.write_times:
            self._field_names[time] = []
            with Dataset(self._file_name(time, suffix)) as data:
                for key in data.variables.keys():
                    if data[key].shape[0] == n_points:
                        self._field_names[time].append(key)
        return self._field_names

    @property
    def vertices(self) -> pt.Tensor:
        return self._mesh_data[:, :3]

    @property
    def weights(self) -> pt.Tensor:
        return self._mesh_data[:, 3]
