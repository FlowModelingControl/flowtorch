import pytest
import torch as pt
from flowtorch.data import PSPDataloader


class PSPTestData:
    def __init__(self):
        self.path = "test/test_data/private_data/S4_preliminary/"
        self.test_file = "225.hdf5"
        self.datasets = ["Zone0000", "Zone0001"]
        self.write_times = dict(
            zip(self.datasets, 
            [['0.000000', '0.000500', '0.001000', '0.001500', '0.002000']]*len(self.datasets))
        )
        self.n_snapshots = dict(
            zip(self.datasets, [4367, 4367])
        )
        self.field_name = dict(
            zip(self.datasets, ["Images"]*2)
        )
        self.shapes = dict(
            zip(self.datasets, [(465, 159), (250, 75)])
        )


@pytest.fixture()
def get_test_data():
    yield PSPTestData()


class TestPSPDataloader:
    def test_select_dataset(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        datasets = get_test_data.datasets
        assert loader._dataset_names == datasets
        assert loader._dataset == datasets[0]
        loader.select_dataset(datasets[1])
        assert loader._dataset == datasets[1]
        loader.select_dataset("Banana")
        assert loader._dataset == datasets[1]

    def test_write_times(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        for dataset in get_test_data.datasets:
            first_times = get_test_data.write_times[dataset]
            loader.select_dataset(dataset)
            assert loader.write_times()[:len(first_times)] == first_times

    def test_field_names(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        for dataset in get_test_data.datasets:
            loader.select_dataset(dataset)
            assert loader.field_names()[0] == get_test_data.field_name[dataset]

    def test_load_snapshot(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        for dataset in get_test_data.datasets:
            sample_time = get_test_data.write_times[dataset][-1]
            field = get_test_data.field_name[dataset]
            loader.select_dataset(dataset)
            snapshot = loader.load_snapshot(field, sample_time)
        assert snapshot.size() == get_test_data.shapes[dataset]

    def test_get_vertices(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        for dataset in get_test_data.datasets:
            loader.select_dataset(dataset)
            vertices = loader.get_vertices()
            shape = get_test_data.shapes[dataset]
            assert tuple(vertices.size()) == (*shape, 3)

    def test_get_weights(self, get_test_data):
        file_path = get_test_data.path + get_test_data.test_file
        loader = PSPDataloader(file_path)
        for dataset in get_test_data.datasets:
            loader.select_dataset(dataset)
            weights = loader.get_weights()
            assert tuple(weights.size()) == get_test_data.shapes[dataset]
