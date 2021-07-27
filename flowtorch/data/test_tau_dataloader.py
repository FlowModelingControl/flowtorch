# standard library packages
import pytest
# third party packages
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import TAUDataloader


class TestTAUDataloader():
    def test_find_grid_file(self):
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, "sol.pval.unsteady_")
        grid_file = loader._find_grid_file()
        assert grid_file == "PW_DES-HybQuadTRex-v2_yp-50_s1.15_ny67.grd"

    def test_decompose_file_name(self):
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, "sol.pval.unsteady_")
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 10
        assert "2.9580000000e-02" in time_iter
        assert time_iter["2.9580000000e-02"] == "14790"

    def test_properties(self):
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, "sol.pval.unsteady_")
        times = loader.write_times
        assert times[0] == "2.9580000000e-02"
        assert times[-1] == "3.2190000000e-02"
        field_names = loader.field_names
        assert len(field_names.keys()) == 10
        assert "density" in field_names[times[-1]]
        n_points = 1119348
        vertices = loader.vertices
        assert vertices.shape == (n_points, 3)
        weights = loader.weights
        assert weights.shape == (n_points,)

    def test_load_snapshot(self):
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, "sol.pval.unsteady_")
        times = loader.write_times
        field_names = loader.field_names[times[-1]]
        n_points = 1119348
        # single snapshot, single field
        field = loader.load_snapshot(field_names[0], times[-1])
        assert field.shape == (n_points,)
        # multiple snapshots, single field
        field_series = loader.load_snapshot(field_names[0], times[-2:])
        assert field_series.shape == (n_points, 2)
        assert pt.allclose(field_series[:, -1], field)
        # single snapshot, multiple fields
        f1, f2 = loader.load_snapshot(field_names[:2], times[-1])
        assert f1.shape == (n_points,)
        assert f2.shape == (n_points,)
        # multiple snapshots, multiple field
        f1s,  f2s = loader.load_snapshot(field_names[:2], times[-2:])
        assert f1s.shape == (n_points, 2)
        assert f2s.shape == (n_points, 2)
        assert pt.allclose(f1s[:, -1], f1)
        assert pt.allclose(f2s[:, -1], f2)
