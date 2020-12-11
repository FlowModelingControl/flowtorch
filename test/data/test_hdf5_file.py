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
        self.test_cases = [
            "of_cavity_ascii",
            "of_cavity_binary",
            "of_cavity_ascii_parallel",
            "of_cavity_binary_parallel"
        ]
        self.const_group = sorted(["vertices", "connectivity", "centers", "volumes"])
        self.var_group = ["0.1", "0.2", "0.3", "0.4", "0.5"]


@pytest.fixture()
def get_test_data():
    yield HDF5TestData()


class TestHDF5Writer():
    def test_write(self, get_test_data):
        file_path = get_test_data.test_path + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write("ones", (3, 2), pt.ones(3, 2), "0.01")
        writer.write("twos", (3, 2), pt.ones(3, 2)*2, "0.01")
        writer.write("threes", (3, 2), pt.ones(3, 2)*3, "0.03")
        del writer
        hdf5_file = File(file_path, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
        assert os.path.isfile(file_path)
        assert list(hdf5_file["variable"].keys()) == ["0.01", "0.03"]
        hdf5_file.close()

    def test_write_const(self, get_test_data):
        file_path = get_test_data.test_path + "test_file.hdf5"
        writer = HDF5Writer(file_path)
        writer.write("zeros", (3, 2), pt.zeros(3,2), dtype=pt.float64)
        writer.write("zeros_single", (3, 2), pt.zeros(3,2), dtype=pt.float32)
        del writer
        hdf5_file = File(file_path, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
        assert list(hdf5_file["constant"].keys()) == ["zeros", "zeros_single"]
        assert hdf5_file["constant/zeros"].dtype == "float64"
        assert hdf5_file["constant/zeros_single"].dtype == "float32"
        hdf5_file.close()

class TestFOAM2HDF5():
    def test_convert(self, get_test_data):
        for case in get_test_data.test_cases:
            case_path = get_test_data.test_path + case
            converter = FOAM2HDF5(case_path)
            converter.convert("flowtorch.hdf5")
            del converter
            filename = case_path + "/flowtorch.hdf5"
            if os.path.isfile(filename):
                hdf5_file = File(filename, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
                const_keys = sorted(hdf5_file["constant"].keys())
                assert const_keys == get_test_data.const_group
                var_keys = sorted(hdf5_file["variable"].keys())
                assert var_keys == get_test_data.var_group
                assert hdf5_file["constant/volumes"].shape[0] == 400
                assert hdf5_file["constant/centers"].shape == (400, 3)
                hdf5_file.close()            