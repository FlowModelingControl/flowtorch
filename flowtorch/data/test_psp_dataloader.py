# standard library packages
import pytest
# third party packages
import torch as pt
# flowtorch packages
from flowtorch.data import PSPDataloader

# this path will be replaced once non-proprietary test data is available
PATH = "/home/andre/Downloads/input/FOR/iPSP_reference_may_2021/0226.hdf5"


class TestPSPDataloader():
    def test_zone(self):
        loader = PSPDataloader(PATH)
        names = loader.zone_names
        assert "Zone0000" in names and "Zone0001" in names
        assert loader.zone == "Zone0000"
        loader.zone = names[-1]
        assert loader.zone == names[-1]
        loader.zone = ""
        assert loader.zone == names[-1]

    def test_info(self):
        loader = PSPDataloader(PATH)
        info = loader.info
        assert "Mach" in info.keys()
        assert len(info["Mach"]) == 2
        assert isinstance(info["Mach"][0], float)
        assert isinstance(info["Mach"][1], str)

    def test_zone_info(self):
        loader = PSPDataloader(PATH)
        info = loader.zone_info
        assert "SamplingFrequency" in info
        assert isinstance(info["SamplingFrequency"][0], float)
        assert info["ZoneName"][0] == "Wing"
        loader.zone = loader.zone_names[-1]
        info = loader.zone_info
        assert info["ZoneName"][0] == "HTP"

    def test_time_to_index(self):
        loader = PSPDataloader(PATH)
        times = loader.write_times
        indices = loader._time_to_index(times)
        assert indices == list(range(4367))

    def test_common_properties(self):
        loader = PSPDataloader(PATH)
        n_snapshots = 4367
        freq = 2000.5
        times = loader.write_times
        assert len(times) == n_snapshots
        assert times[-1] == str(round((n_snapshots - 1) / freq, 8))
        assert times[0] == str(round(0.0, 8))
        fields = loader.field_names
        assert len(fields.keys()) == 1
        assert fields[times[0]][0] == "Cp"
        vertices = loader.vertices
        assert vertices.shape == (465, 159, 3)
        weights = loader.weights
        assert weights.shape == (465, 159)
        loader.zone = loader.zone_names[-1]
        vertices = loader.vertices
        assert vertices.shape == (250, 75, 3)
        weights = loader.weights
        assert weights.shape == (250, 75)
        # load single field, single snapshot
        cp = loader.load_snapshot("Cp", times[-1])
        assert cp.shape == (250, 75)
        # load single field, multiple snapshots
        cp_s = loader.load_snapshot("Cp", times[-10:])
        assert cp_s.shape == (250, 75, 10)
        assert pt.allclose(cp, cp_s[:, :, -1])
        # load multiple fields, single snapshot
        cp, = loader.load_snapshot(["Cp"], times[0])
        assert cp.shape == (250, 75)
        # load multiple fields, multiple snapshots
        cp_s, = loader.load_snapshot(["Cp"], times[:10])
        assert pt.allclose(cp, cp_s[:, :, 0])
