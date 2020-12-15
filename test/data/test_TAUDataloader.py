import pytest
import torch as pt
from flowtorch.data import TAUDataloader


class TAUTestData:
    def __init__(self):
        self.path = "test/test_data/external_data/tau_backward_facing_step/"


@pytest.fixture()
def get_test_data():
    yield TAUTestData()


class TestTAUDataloader:
    def test_write_times(self, get_test_data):
        pass

    def test_field_names(self, get_test_data):
        pass

    def test_load_snapshot(self, get_test_data):
        pass

    def test_get_vertices(self, get_test_data):
        pass

    def test_get_weights(self, get_test_data):
        pass