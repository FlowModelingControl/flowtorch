import os
import pytest
import sys
import torch as pt
from h5py import File
from mpi4py import MPI
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
        writer.write(pt.ones(3, 2)*2, "twos", "0.01")
        writer.write(pt.ones(3, 2)*3, "threes", "0.03")
        del writer
        hdf5_file = File(file_path, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
        assert os.path.isfile(file_path)
        assert list(hdf5_file["variable"].keys()) == ["0.01", "0.03"]
        hdf5_file.close()

    def test_write_const(self, get_test_data):
        file_path = get_test_data.test_path + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write_const(pt.zeros(3,2), "zeros", dtype=pt.float64)
        writer.write_const(pt.zeros(3,2), "zeros_single", dtype=pt.float32)
        del writer
        hdf5_file = File(file_path, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
        assert list(hdf5_file["constant"].keys()) == ["zeros", "zeros_single"]
        assert hdf5_file["constant/zeros"].dtype == "float64"
        assert hdf5_file["constant/zeros_single"].dtype == "float32"
        hdf5_file.close()