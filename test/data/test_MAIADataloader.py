import pytest
import torch as pt
from flowtorch.data import MAIADataloader


class MAIATestData:
    def __init__(self):
        self.path = "test/test_data/external_data/maia_cylinder/"


@pytest.fixture()
def get_test_data():
    yield MAIATestData()


class TestMAIADataloader:
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