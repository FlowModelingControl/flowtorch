"""Module to load FOR2895 iPSP data.

The :class:`PSPDataloader` class allows to load
instationary pressure-sensitive paint (iPSP_) data provided by
DLR (Deutsches Luft- und Raumfahrtzentrum) within the FOR2895
research group.

.. _iPSP: https://www.dlr.de/as/en/desktopdefault.aspx/tabid-183/251_read-13334/
"""

# standard library packages
from math import ceil
from os.path import exists
from typing import List, Dict, Union
import sys
# third party packages
from h5py import File
import numpy as np
import torch as pt
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_list_or_str


COORDINATE_KEYS = ["CoordinatesX", "CoordinatesY", "CoordinatesZ"]
INFO_KEY = "Info"
PARAMETER_KEY = "Parameter"
DESCRIPTION_KEY = "ParameterDescription"
WEIGHT_KEY = "Mask"
TIME_KEY = "TimeValues"
FIELDS = {
    "Cp": "Images"
}
FREQUENCY_KEY = "SamplingFrequency"


class PSPDataloader(Dataloader):
    """Load iPSP data and meta data.

    iPSP data comes as an HDF5 file with datasets organized in
    different zones. Each zone has additional meta data related
    to flow conditions and camera setting. The active zone may be
    switched by seeting the `zone` attribute.

    Examples

    >>> from flowtorch import PSPDataloader
    >>> loader = PSPDataloader("0226.hdf5")
    >>> loader.zone_names
    ['Zone0000', 'Zone0001']
    >>> loader.info.keys()
    ['AngleAttackAlpha', 'DateOfRecording', 'Mach', ...]
    >>> loader.info["Mach"]
    (0.903, 'Mach number')
    >>> loader.zone_info.keys()
    ['ExposureTime', 'NumberImages', 'PSPDeviceName', 'SamplingFrequency', 'ZoneName']
    >>> loader.zone
    Zone0000
    >>> loader.zone = "Zone0001"
    >>> loader.zone_info["ZoneName"]
    HTP
    >>> cp = loader.load_snapshot("Cp", loader.write_times[:10])
    >>> cp.shape
    torch.Size([250, 75, 10])

    """

    def __init__(self, path: str, dtype: str = DEFAULT_DTYPE):
        """Create PSPDataloader instance from file path.

        :param path: path to iPSP file
        :type path: str
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._path = path
        self._dtype = dtype
        if exists(self._path):
            self._file = File(self._path, mode="r")
        else:
            raise FileNotFoundError(f"Could not find file {path}")
        self._zone_names = None
        self._zone = self.zone_names[0]
        self._info = None

    def _time_to_index(self, time: Union[List[str], str]) -> Union[List[int], int]:
        freq = self.zone_info[FREQUENCY_KEY][0]
        if isinstance(time, list):
            return [int(round(float(t) * freq, 0)) for t in time]
        else:
            return int(round(float(time) * freq, 0))

    def _load_single_field(self, field_name: str, ind: Union[np.ndarray, int]) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{self._zone}/{FIELDS[field_name]}"][:, :, ind],
            dtype=self._dtype
        )

    def load_snapshot(self, field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        ind = self._time_to_index(time)
        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [
                    self._load_single_field(name, np.array(ind)) for name in field_name
                ]
            else:
                return [
                    self._load_single_field(name, ind) for name in field_name
                ]
        # load single field
        else:
            if isinstance(time, list):
                return self._load_single_field(field_name, np.array(ind))
            else:
                return self._load_single_field(field_name, ind)

    @property
    def zone_names(self) -> List[str]:
        if self._zone_names is None:
            keys = self._file.keys()
            self._zone_names = [key for key in keys if key.startswith("Zone")]
            if len(self._zone_names) < 1:
                raise ValueError(f"No valid zones in file {self._path}")
        return self._zone_names

    @property
    def zone(self) -> str:
        return self._zone

    @zone.setter
    def zone(self, zone_name: str):
        if zone_name in self._zone_names:
            self._zone = zone_name
        else:
            print(f"{zone_name} not found. Available zones are:")
            print(self._zone_names)

    @property
    def info(self) -> Dict[str, tuple]:
        if self._info is None:
            parameters = self._file[f"{INFO_KEY}/{PARAMETER_KEY}"].attrs
            descriptions = self._file[f"{INFO_KEY}/{DESCRIPTION_KEY}"].attrs
            self._info = dict()
            for key in parameters.keys():
                self._info[key] = (
                    parameters.get(key, ""), descriptions.get(key, "")
                )
        return self._info

    @property
    def zone_info(self) -> Dict[str, tuple]:
        parameters = self._file[f"{self._zone}/{PARAMETER_KEY}"].attrs
        descriptions = self._file[f"{self._zone}/{DESCRIPTION_KEY}"].attrs
        self._zone_info = dict()
        for key in parameters.keys():
            self._zone_info[key] = (
                parameters.get(key, ""), descriptions.get(key, "")
            )
        return self._zone_info

    @property
    def write_times(self) -> List[str]:
        times = self._file[f"{self._zone}/{TIME_KEY}"][:]
        return [str(round(t, 8)) for t in times]

    @property
    def field_names(self) -> Dict[str, List[str]]:
        return {self.write_times[0]: list(FIELDS.keys())}

    @property
    def vertices(self) -> pt.Tensor:
        return pt.stack([pt.tensor(
            self._file[f"{self.zone}/{coord}"][:, :], dtype=self._dtype
        ) for coord in COORDINATE_KEYS], dim=-1)

    @property
    def weights(self) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{self.zone}/{WEIGHT_KEY}"][:, :], dtype=self._dtype
        )
