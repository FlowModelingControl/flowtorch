# standard library packages
from os.path import join

# third party packages
import pytest
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import TAUDataloader


class TestTAUDataloader():
    def test_get_domain_ids(self):
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        assert loader._domain_ids == ["249", "258"]

    def test_find_grid_file(self):
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
        grid_file = loader._find_grid_file()
        assert grid_file == "PW_DES-HybQuadTRex-v2_yp-50_s1.15_ny67.grd"

    def test_decompose_file_name(self):
        # serial test case
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 2
        assert "2.9580000000e-02" in time_iter
        assert time_iter["2.9580000000e-02"] == "14790"
        # distributed test case
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 1
        assert "3.409440000e-01" in time_iter
        assert time_iter["3.409440000e-01"] == "17508"

    def test_load_domain_mesh_data(self):
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        data = loader._load_domain_mesh_data("249")
        assert data.shape == (17382, 4)
        data = loader._load_domain_mesh_data("258")
        assert data.shape == (17128, 4)

    def test_load_mesh_data(self):
        # serial/reconstructed
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
        n_points = 1119348
        assert loader.vertices.shape == (n_points, 3)
        assert loader.weights.shape == (n_points,)
        # distributed
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        n_points = 34510
        assert loader.vertices.shape == (n_points, 3)
        assert loader.weights.shape == (n_points,)

    def test_write_times(self):
        # serial/reconstructed
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
        times = loader.write_times
        assert times[0] == "2.9580000000e-02"
        assert times[-1] == "3.2190000000e-02"
        # distributed
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        assert loader.write_times[0] == "3.409440000e-01"

    def test_field_names(self):
        # serial/reconstructed
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
        times = loader.write_times
        assert len(loader.field_names.keys()) == 2
        assert "pressure" in loader.field_names[times[-1]]
        assert "density" in loader.field_names[times[-1]]
        # distributed
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        times = loader.write_times
        assert len(loader.field_names.keys()) == 1
        assert "pressure" in loader.field_names[times[0]]
        assert "density" in loader.field_names[times[0]]

    def test_load_snapshot(self):
        # serial/reconstructed
        path = DATASETS["tau_backward_facing_step"]
        loader = TAUDataloader(path, path, "sol.pval.unsteady_")
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
        # distributed
        path = DATASETS["tau_distributed"]
        prefix = "solution_Delta_MUB_fine_iso_hybrid_58.pval.unsteady_"
        loader = TAUDataloader(path, join(path, "mesh"), prefix,
                               distributed=True, subfolders=True)
        times = loader.write_times
        n_points = 34510
        # single snapshots, single field
        density = loader.load_snapshot("density", times[0])
        assert density.shape == (n_points,)
        # multiple snapshots, single field
        density = loader.load_snapshot("density", [times[0], times[0]])
        assert density.shape == (n_points, 2)
        # single snapshot, multiple fields
        density, p = loader.load_snapshot(
            ["density", "pressure"], times[0])
        assert density.shape == (n_points, )
        assert p.shape == (n_points, )
        # multiple snapshots, multiple fields
        density, p = loader.load_snapshot(["density", "pressure"],
                                          [times[0], times[0]])
        assert density.shape == (n_points, 2)
        assert p.shape == (n_points, 2)
