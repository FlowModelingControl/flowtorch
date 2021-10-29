r"""Direct access to TAU simulation data.

The DRL (Deutsches Luft- und Raumfahrtzentrum) TAU_ code saves
snapshots in the NetCFD format. The :class:`TAUDataloader` is a
wrapper around the NetCFD Python bindings to simplify the access
to snapshot data.

.. _TAU: https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/

"""
# standard library packages
from glob import glob
from typing import List, Dict, Tuple, Union
# third party packages
from netCDF4 import Dataset
import torch as pt
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_list_or_str, check_and_standardize_path


IGNORE_FIELDS = ("x", "y", "z", "volume", "global_id")
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
    >>> loader = TAUDataloader(path, base_name="sol.pval.unsteady_")
    >>> times = loader.write_times
    >>> fields = loader.field_names[times[0]]
    >>> fields
    ['density', 'x_velocity', 'y_velocity', ...]
    >>> density = loader.load_snapshot("density", times)
    >>> density.shape
    torch.Size([1119348, 10])

    """

    def __init__(self, path: str, base_name: str, dtype: str = DEFAULT_DTYPE):
        """Create loader instance from TAU simulation folder.

        :param path: path to TAU simulation files
        :type path: str
        :param base_name: part of the solution file name before iteration count,
            e.g., base_name_ if the solution file is called base_name_i=0102_t=1.0
        :type base_name: str
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._path = check_and_standardize_path(path)
        self._base_name = base_name
        self._dtype = dtype
        self._time_iter = self._decompose_file_name()

    def _find_grid_file(self) -> str:
        """Determine the name of the grid file

        :raises FileNotFoundError: if no grid file is found
        :raises FileNotFoundError: if multiple grid files are found
        :return: name of the grid file
        :rtype: str
        """
        files = glob(f"{self._path}/*.grd")
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find mesh file (.grd) in {self._path}/")
        if len(files) > 1:
            raise FileNotFoundError(
                f"Found multiple mesh files (.grd) in {self._path}/")
        return files[0].split("/")[-1]

    def _decompose_file_name(self) -> Dict[str, str]:
        """Extract write time and iteration from file name.

        :raises FileNotFoundError: if no solution files are found
        :return: dictionary with write times as keys and the corresponding
            iterations as values
        :rtype: Dict[str, str]
        """
        files = glob(f"{self._path}/{self._base_name}i=*t=*")
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find solution files in {self._path}/")
        time_iter = {}
        for f in files:
            t = f.split("t=")[-1]
            i = f.split("i=")[-1].split("_t=")[0]
            time_iter[t] = i
        return time_iter

    def _file_name(self, time: str) -> str:
        """Create solution file name from write time.

        :param time: snapshot write time
        :type time: str
        :return: name of solution file
        :rtype: str
        """
        itr = self._time_iter[time]
        return f"{self._path}/{self._base_name}i={itr}_t={time}"

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load a single snapshot of a single field from the netCDF4 file.

        :param field_name: name of the field
        :type field_name: str
        :param time: snapshot write time
        :type time: str
        :return: tensor holding the field values
        :rtype: pt.Tensor
        """
        path = self._file_name(time)
        with Dataset(path) as data:
            field = pt.tensor(data.variables[field_name][:], dtype=self._dtype)
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
        self._field_names = {}
        for time in self.write_times:
            self._field_names[time] = []
            with Dataset(self._file_name(time)) as data:
                self._field_names[time] = [
                    key for key in data.variables.keys() if key not in IGNORE_FIELDS
                ]
        return self._field_names

    @property
    def vertices(self) -> pt.Tensor:
        path = f"{self._path}/{self._find_grid_file()}"
        with Dataset(path) as data:
            vertices = pt.stack(
                [pt.tensor(data[key][:], dtype=self._dtype)
                 for key in VERTEX_KEYS],
                dim=-1
            )
        return vertices

    @property
    def weights(self) -> pt.Tensor:
        path = self._file_name(self.write_times[0])
        with Dataset(path) as data:
            if WEIGHT_KEY in data.variables.keys():
                weights = pt.tensor(
                    data.variables[WEIGHT_KEY][:], dtype=self._dtype)
            else:
                print(f"Warning: cell volumes not found in file {path}")
                weights = None
        return weights
