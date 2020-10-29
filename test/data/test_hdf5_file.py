import os
import pytest
import sys
import torch as pt
from h5py import File
from flowtorch.data import HDF5Dataloader, HDF5Writer, FOAM2HDF5

class HDF5TestData:
    def __init__(self):
        self.test_path = "test/test_data/run/"


@pytest.fixture()
def get_test_data():
    yield HDF5TestData()


class TestHDF5Writer():
    def test_write(self, get_test_data):
        file_path = get_test_data.test_path + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write(pt.ones(3, 2), "ones", "0.01")
        assert os.path.isfile(file_path)

    def test_write_const(self, get_test_data):
        pass