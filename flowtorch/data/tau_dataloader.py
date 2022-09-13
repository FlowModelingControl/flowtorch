r"""Direct access to TAU simulation data.

The DRL (Deutsches Luft- und Raumfahrtzentrum) TAU_ code saves
snapshots in the NetCFD format. The :class:`TAUDataloader` is a
wrapper around the NetCFD Python bindings to simplify the access
to snapshot data.

.. _TAU: https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/

"""
# standard library packages
from os.path import join, split
from glob import glob
from typing import List, Dict, Tuple, Union, Set
# third party packages
from netCDF4 import Dataset
import torch as pt
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_list_or_str, check_and_standardize_path

VOL_SOLUTION_NAME = ".pval.unsteady_"
PSOLUTION_POSTFIX = ".domain_"
PMESH_NAME = "domain_{:s}_grid_1"
PVERTEX_KEY = "pcoord"
PWEIGHT_KEY = "pvolume"
PADD_POINTS_KEY = "addpoint_idx"
PGLOBAL_ID_KEY = "globalidx"
VERTEX_KEYS = ("points_xc", "points_yc", "points_zc")
WEIGHT_KEY = "volume"

COMMENT_CHAR = "#"
CONFIG_SEP = ":"
SOLUTION_PREFIX_KEY = "solution_prefix"
GRID_FILE_KEY = "primary_grid"
GRID_PREFIX_KEY = "grid_prefix"
N_DOMAINS_KEY = "n_domains"


class TAUConfig(object):
    """Load and parse TAU parameter files.

    The class does not parse the full content of the parameter file
    but only content that is absolutely needed to load snapshot data.

    """

    def __init__(self, file_path: str):
        """Create a `TauConfig` instance from the file path.

        :param file_path: path to the parameter file
        :type path: str
        """
        self._path, self._file_name = split(file_path)
        with open(join(self._path, self._file_name), "r") as config:
            self._file_content = config.readlines()
        self._config = None

    def _parse_config(self, parameter: str) -> str:
        """Extract a value based on a given pattern.

        Every line of the parameter file follows the structure:
            parameter : value
        This function extracts the value as string and remove potential
        white spaces or comments (#). The separator is expected to be a
        colon.

        :param parameter: the parameter of which to extract the value
        :type pattern: str
        :return: extracted value or empty string
        :rtype: str
        """
        for line in self._file_content:
            if parameter in line:
                return line.split(CONFIG_SEP)[-1].split(COMMENT_CHAR)[0].strip()
        return ""

    def _gather_config(self):
        """Gather all required configuration values.
        """
        config = {}
        config[SOLUTION_PREFIX_KEY] = self._parse_config("Output files prefix")
        config[GRID_FILE_KEY] = self._parse_config("Primary grid filename")
        config[GRID_PREFIX_KEY] = self._parse_config("Grid prefix")
        config[N_DOMAINS_KEY] = int(self._parse_config("Number of domains"))
        self._config = config

    @property
    def path(self) -> str:
        return self._path

    @property
    def config(self) -> dict:
        if self._config is None:
            self._gather_config()
        return self._config


class TAUDataloader(Dataloader):
    """Load TAU simulation data.

    The loader is currently limited to read:
    - internal field solution, serial/reconstructed and distributed
    - mesh vertices, serial and distributed
    - cell volumes, serial (if present) and distributed

    Examples

    >>> from os.path import join
    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import TAUDataloader
    >>> path = DATASETS["tau_backward_facing_step"]
    >>> loader = TAUDataloader(join(path, "simulation.para"))
    >>> times = loader.write_times
    >>> fields = loader.field_names[times[0]]
    >>> fields
    ['density', 'x_velocity', 'y_velocity', ...]
    >>> density = loader.load_snapshot("density", times)

    To load distributed simulation data, set `distributed=True`
    >>> path = DATASETS["tau_cylinder_2D"]
    >>> loader = TAUDataloader(join(path, "simulation.para"), distributed=True)
    >>> vertices = loader.vertices

    """

    def __init__(self, parameter_file: str, distributed: bool = False,
                 dtype: str = DEFAULT_DTYPE):
        """Create loader instance from TAU parameter file.

        :param parameter_file: path to TAU simulation parameter file
        :type parameter_file: str
        :param distributed: True if mesh and solution are distributed in domain
            files; defaults to False
        :type distributed: bool, optional
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._para = TAUConfig(parameter_file)
        self._distributed = distributed
        self._dtype = dtype
        self._time_iter = self._decompose_file_name()
        self._mesh_data = None

    def _decompose_file_name(self) -> Dict[str, str]:
        """Extract write time and iteration from file name.

        :raises FileNotFoundError: if no solution files are found
        :return: dictionary with write times as keys and the corresponding
            iterations as values
        :rtype: Dict[str, str]
        """
        base = join(self._para.path, self._para.config[SOLUTION_PREFIX_KEY])
        base += VOL_SOLUTION_NAME
        suffix = f"{PSOLUTION_POSTFIX}0" if self._distributed else "e???"
        files = glob(f"{base}i=*t=*{suffix}")
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find solution files in {self._sol_path}/")
        time_iter = {}
        split_at = PSOLUTION_POSTFIX if self._distributed else " "
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
        path = join(self._para.path, self._para.config[SOLUTION_PREFIX_KEY])
        return f"{path}{VOL_SOLUTION_NAME}i={itr}_t={time}{suffix}"

    def _load_domain_mesh_data(self, pid: str) -> pt.Tensor:
        """Load vertices and volumes for a single processor domain.

        :param pid: domain id
        :type pid: str
        :return: tensor of size n_points x 4, where n_points is the number
            of unique cells in the domain, and the 4 columns contain the
            coordinates of the vertices (x, y, z) and the cell volumes
        :rtype: pt.Tensor
        """
        prefix = self._para.config[GRID_PREFIX_KEY]
        name = PMESH_NAME.format(pid)
        if not (prefix == "(none)"):
            name = f"{prefix}_{name}"
        path = join(self._para.path, name)
        with Dataset(path) as data:
            vertices = pt.tensor(data[PVERTEX_KEY][:], dtype=self._dtype)
            volumes = pt.tensor(data[PWEIGHT_KEY][:], dtype=self._dtype)
            global_ids = pt.tensor(data[PGLOBAL_ID_KEY][:], dtype=pt.int64)
            n_add_points = data[PADD_POINTS_KEY].shape[0]

        n_points = volumes.shape[0] - n_add_points
        data = pt.zeros((n_points, 4), dtype=self._dtype)
        sorting = pt.argsort(global_ids[:n_points])
        data[:, 0] = vertices[:n_points, 0][sorting]
        data[:, 1] = vertices[:n_points, 1][sorting]
        data[:, 2] = vertices[:n_points, 2][sorting]
        data[:, 3] = volumes[:n_points][sorting]
        return data

    def _load_mesh_data(self):
        """Load mesh vertices and cell volumes.

        The mesh data is saved as class member `_mesh_data`. The tensor has the
        dimension n_points x 4; the first three columns correspond to the x/y/z
        coordinates, and the 4th column contains the volumes.
        """
        if self._distributed:
            n = self._para.config[N_DOMAINS_KEY]
            self._mesh_data = pt.cat(
                [self._load_domain_mesh_data(str(pid)) for pid in range(n)],
                dim=0
            )
        else:
            path = join(self._para.path, self._para.config[GRID_FILE_KEY])
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
            self._mesh_data = pt.cat((vertices, weights.unsqueeze(-1)), dim=-1)

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
            for pid in range(self._para.config[N_DOMAINS_KEY]):
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
            n_points = self._load_domain_mesh_data("0").shape[0]
            suffix = ".domain_0"
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
        if self._mesh_data is None:
            self._load_mesh_data()
        return self._mesh_data[:, :3]

    @property
    def weights(self) -> pt.Tensor:
        if self._mesh_data is None:
            self._load_mesh_data()
        return self._mesh_data[:, 3]
