import pytest
import torch as pt
from flowtorch.data import PSPDataloader


class PSPTestData:
    def __init__(self):
        self.path = "test/test_data/private_data/MK1_iPSP_raw_examples/iPSP/"
        self.test_files = [
            "225.hdf5", "227.hdf5", "403.hdf5"
        ]
        datasets = ["HTP", "Wing"]
        self.datasets = dict(
            zip(self.test_files, [datasets] * len(self.test_files))
        )
        self.write_times = {
            "225.hdf5": ['0.000000', '0.001000', '0.002000', '0.003000', '0.004000'],
            "227.hdf5": ['0.000000', '0.001000', '0.002000', '0.003000', '0.004000'],
            "403.hdf5": ['0.000000', '0.001000', '0.002000', '0.003000', '0.004000']
        }
        self.field_names = {}
        field_list = ["cp"]
        for case in self.test_files:
            field_names = dict(
                zip(self.write_times[case], [field_list]
                    * len(self.write_times[case]))
            )
            self.field_names[case] = field_names


@pytest.fixture()
def get_test_data():
    yield PSPTestData()


class TestPSPDataloader:
    def test_select_dataset(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            datasets = get_test_data.datasets[case]
            assert loader._dataset_names == datasets
            assert loader._dataset == datasets[0]
            loader.select_dataset(datasets[1])
            assert loader._dataset == datasets[1]
            loader.select_dataset("Banana")
            assert loader._dataset == datasets[1]

    def test_write_times(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            first_times = get_test_data.write_times[case]
            assert loader.write_times()[:len(first_times)] == first_times

    def test_field_names(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            first_times = get_test_data.field_names[case]
            assert list(loader.field_names().keys())[
                :len(first_times)] == list(first_times.keys())
            assert list(loader.field_names().values())[
                :len(first_times)] == list(first_times.values())

    def test_load_snapshot(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            sample_time = get_test_data.write_times[case][-1]
            snapshot = loader.load_snapshot("cp", sample_time)
            assert snapshot.size()[0] == 18750
            loader.select_dataset("Wing")
            snapshot = loader.load_snapshot("cp", sample_time)
            assert snapshot.size()[0] == 73935

    def test_get_vertices(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            vertices = loader.get_vertices()
            assert tuple(vertices.size()) == (18750, 3)
            loader.select_dataset("Wing")
            vertices = loader.get_vertices()
            assert tuple(vertices.size()) == (73935, 3)
            vertices = loader.get_vertices(10000, 20000)
            assert tuple(vertices.size()) == (10000, 3)
            vertices = loader.get_vertices(70000, 80000)
            assert tuple(vertices.size()) == (3935, 3)

    def test_get_weights(self, get_test_data):
        for case in get_test_data.test_files:
            file_path = get_test_data.path + case
            loader = PSPDataloader(file_path)
            weights = loader.get_weights()
            assert weights.size()[0] == 18750
            loader.select_dataset("Wing")
            weights = loader.get_weights()
            assert weights.size()[0] == 73935
