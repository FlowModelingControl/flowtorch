"""Global constants for all subpackages."""

from os import environ, listdir
from os.path import isdir, join
from torch import float32

# default data type for all PyTorch tensors
DEFAULT_DTYPE = float32
# default tolerance for comparison of floating point numbers
FLOAT_TOLERANCE = 1.0e-7
# path to flowTorch datasets
DATASET_PATH = environ.get("FLOWTORCH_DATASETS", "")
assert DATASET_PATH != "", "FLOWTORCH_DATASETS environment variable not defined."
# available datasets
folders = [folder for folder in listdir(DATASET_PATH) if isdir(join(DATASET_PATH, folder))]
folder_paths = [join(DATASET_PATH, folder) for folder in folders]
DATASETS = dict(zip(folders, folder_paths))