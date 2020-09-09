import os
import pytest
import sys
import torch as pt
from flowtorch.data import FOAMDataloader, FOAMCase


class FOAMTestData:
    def __init__(self):
        self.data_path = "test/test_data/run/"
        self.test_cases = [
            "cavity_ascii",
            "cavity_binary",
            "cavity_ascii_parallel",
            "cavity_binary_parallel"
        ]
        self.distributed = dict(
            zip(self.test_cases, [False, False, True, True])
        )
        self.processors = dict(
            zip(self.test_cases, [1, 1, 4, 4])
        )
        times = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        self.times = dict(
            zip(self.test_cases, [times] * len(self.test_cases))
        )
        self.paths = dict(
            zip(self.test_cases,
                [self.data_path + case for case in self.test_cases])
        )
        self.file_paths = {}
        for case in self.test_cases:
            if self.distributed[case]:
                self.file_paths[case] = [
                    self.paths[case] + "/processor0/{:s}/U".format(time)
                    for time in self.times[case]
                ]
            else:
                self.file_paths[case] = [
                    self.paths[case] + "/{:s}/U".format(time)
                    for time in self.times[case]
                ]

        for key in self.paths.keys():
            path = self.paths[key]
            if not os.path.exists(path):
                sys.exit("Error: could not find test case {:s}".format(path))
        field_names = {'0': ['U', 'p'],
                       '0.1': ['U', 'p', 'phi'],
                       '0.2': ['U', 'p', 'phi'],
                       '0.3': ['U', 'p', 'phi'],
                       '0.4': ['U', 'p', 'phi'],
                       '0.5': ['U', 'p', 'phi']
                       }
        self.field_names = dict(
            zip(self.test_cases, [field_names] * len(self.test_cases))
        )


@pytest.fixture()
def get_test_data():
    yield FOAMTestData()


class TestFOAMCase:
    def test_distributed(self, get_test_data):
        for key in get_test_data.distributed.keys():
            case = FOAMCase(get_test_data.paths[key])
            distributed = get_test_data.distributed[key]
            assert distributed == case.distributed()

    def test_processors(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            assert get_test_data.processors[key] == case.processors()

    def test_build_file_path(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            file_paths_test = get_test_data.file_paths[key]
            file_paths = [
                case.build_file_path("U", time, 0)
                for time in get_test_data.times[key]
            ]
            assert file_paths_test == file_paths

    def test_write_times(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            write_times = get_test_data.times[key]
            assert write_times == case.write_times()

    def test_field_names(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = FOAMCase(get_test_data.paths[key])
            field_names = get_test_data.field_names[key]
            assert field_names == case.field_names()


class TestFOAMDataloader:
    def test_instantiation(self, get_test_data):
        for key in get_test_data.paths.keys():
            case = get_test_data.paths[key]
            FOAMDataloader(case)
