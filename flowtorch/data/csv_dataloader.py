"""Dataloader and accompanying tools to work with CSV files.

A lot of scientific data is exchanged as comma separated value (CSV) files.
While there are many Python packages available to read such data, one has
to understand how the data is organized in the CSV file before being able
to use the readers properly. Moreover, time series data sometimes come as
individual files in a single folder or as time folders with the respective
snapshot data inside that folder. This subpackages simplifies access to
common CSV-based time series data by trying to figure out appropriate reader
settings automatically. 
"""

# standard library packages
from glob import glob
from typing import List, Dict, Tuple
# third party packages
import torch as pt
from pandas import read_csv, DataFrame
# flowtorch packages
from .dataloader import Dataloader
from .utils import check_and_standardize_path

VERTEX_KEY = "vertices"
WEIGHT_KEY = "weights"
FIELD_KEY = "fields"

DAVIS_KEYS = {
    VERTEX_KEY: ["x", "y"],
    WEIGHT_KEY: "isValid",
    FIELD_KEY: ["Vx", "Vy", "Vz"]
}

PANDAS_SKIPROWS = "skiprows"
PANDAS_SEP = "sep"
PANDAS_HEADER = "header"
PANDAS_NAMES = "names"
PANDAS_ARGS = [PANDAS_SKIPROWS, PANDAS_SEP, PANDAS_HEADER, PANDAS_NAMES]


class CSVDataloader(Dataloader):
    def __init__(self, path: str, prefix: str, suffix: str, read_options: dict,
                 time_folders: bool, dtype=pt.float32):
        """
        """
        self._path = check_and_standardize_path(path)
        self._prefix = prefix
        self._suffix = suffix
        self._read_options = read_options
        self._time_folders = time_folders
        self._dtype = dtype
        self._write_times = None

    @classmethod
    def from_davis(cls, path: str, prefix: str = "", suffix: str = ".dat", dtype=pt.float32):
        read_options = {
            PANDAS_SKIPROWS: [0, 1, 2],
            PANDAS_HEADER: None,
            PANDAS_SEP: " ",
            PANDAS_NAMES: DAVIS_KEYS[VERTEX_KEY] + DAVIS_KEYS[FIELD_KEY] + [DAVIS_KEYS[WEIGHT_KEY]],
            VERTEX_KEY: DAVIS_KEYS[VERTEX_KEY],
            WEIGHT_KEY: DAVIS_KEYS[WEIGHT_KEY],
            FIELD_KEY: DAVIS_KEYS[FIELD_KEY]
        }
        return cls(path, prefix, suffix, read_options, False, dtype)

    def _build_file_path(self, time: str) -> str:
        if self._time_folders:
            return f"{self._path}/{time}/{self._suffix}{self._prefix}"
        else:
            return f"{self._path}/{self._prefix}{time}{self._suffix}"

    def _load_csv(self, time: str) -> DataFrame:
        file_path = f"{self._path}/{self._prefix}{time}{self._suffix}"
        options = {key: self._read_options[key] for key in PANDAS_ARGS}
        return read_csv(file_path, **options)

    def write_times(self) -> List[str]:
        if self._write_times is None:
            if self._time_folders:
                folders = glob(f"{self._path}/*")
                self._write_times = sorted(
                    [folder.split("/")[-1] for folder in folders], key=float
                )
            else:
                files = glob(self._build_file_path("*"))
                self._write_times = sorted(
                    [f.split("/")[-1][len(self._prefix):-len(self._suffix)]
                     for f in files], key=float
                )
        return self._write_times

    def field_names(self) -> Dict[str, List[str]]:
        return dict(
            {self.write_times()[0] : self._read_options[FIELD_KEY]}
        )

    def load_snapshot(self, field_name: List[str], time: str) -> pt.Tensor:
        """
        """
        snapshot = self._load_csv(time)
        return pt.tensor(
            snapshot[field_name].values, dtype=self._dtype
        )

    def get_vertices(self) -> pt.Tensor:
        snapshot = self._load_csv(self._write_times[0])
        return pt.tensor(
            snapshot[self._read_options[VERTEX_KEY]].values, dtype=self._dtype
        )

    def get_weights(self) -> pt.Tensor:
        snapshot = self._load_csv(self._write_times[0])
        return pt.tensor(
            snapshot[self._read_options[WEIGHT_KEY]].values, dtype=self._dtype
        )
