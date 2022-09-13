# standard library packages
from os.path import join

# third party packages
import pytest
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import TAUDataloader, TAUConfig


SOLUTION_PREFIX_KEY = "solution_prefix"
GRID_FILE_KEY = "primary_grid"
GRID_PREFIX_KEY = "grid_prefix"
N_DOMAINS_KEY = "n_domains"


class TestTAUConfig():
    path_0 = DATASETS["tau_backward_facing_step"]
    path_1 = DATASETS["tau_cylinder_2D"]
    file_name = "simulation.para"

    def test_init(self):
        config = TAUConfig(join(self.path_0, self.file_name))
        assert config._path == self.path_0
        assert config._file_name == self.file_name
        assert len(config._file_content) > 0
        assert all([isinstance(line, str) for line in config._file_content])
        config = TAUConfig(join(self.path_1, self.file_name))
        assert config._path == self.path_1
        assert config._file_name == self.file_name

    def test_parse_config(self):
        config = TAUConfig(join(self.path_0, self.file_name))
        empty = config._parse_config("Not in the File")
        assert empty == ""
        relaxation_solver = config._parse_config("Relaxation solver")
        assert relaxation_solver == "Runge_Kutta"
        n_mg = config._parse_config("Number of multigrid levels")
        assert n_mg == "1"

    def test_gather_config(self):
        # backward facing step
        config = TAUConfig(join(self.path_0, self.file_name))
        config_values = config.config
        assert config_values[SOLUTION_PREFIX_KEY] == "sol_files/sol"
        assert config_values[GRID_FILE_KEY] == "grid_files/PW_DES-HybQuadTRex-v2_yp-50_s1.15_ny67.grd"
        assert config_values[GRID_PREFIX_KEY] == "grid_files/distributed/dual"
        assert config_values[N_DOMAINS_KEY] == 96
        # 2D flow past a cylinder
        config = TAUConfig(join(self.path_1, self.file_name))
        config_values = config.config
        assert config_values[SOLUTION_PREFIX_KEY] == "solution/solution"
        assert config_values[GRID_FILE_KEY] == "cylinder_scaled.grid"
        assert config_values[GRID_PREFIX_KEY] == "dualgrid/dual"
        assert config_values[N_DOMAINS_KEY] == 16


class TestTAUDataloader():
    path_0 = DATASETS["tau_backward_facing_step"]
    path_1 = DATASETS["tau_cylinder_2D"]
    file_name = "simulation.para"

    def test_decompose_file_name(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 1
        assert "4.69800000000e-02" in time_iter
        assert time_iter["4.69800000000e-02"] == "23490"
        # backward facing step, distributed
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 1
        assert "4.69800000000e-02" in time_iter
        assert time_iter["4.69800000000e-02"] == "23490"
        # 2D cylinder, serial
        loader = TAUDataloader(join(self.path_1, self.file_name))
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 1
        assert "6.0000e+02" in time_iter
        assert time_iter["6.0000e+02"] == "1000"
        # 2D cylinder, distributed
        loader = TAUDataloader(join(self.path_1, self.file_name), True)
        time_iter = loader._decompose_file_name()
        assert len(time_iter.keys()) == 1
        assert "6.0000e+02" in time_iter
        assert time_iter["6.0000e+02"] == "1000"

    def test_file_name(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        file_name = join(self.path_0, "sol_files/sol") + \
            ".pval.unsteady_i=23490_t=4.69800000000e-02"
        assert loader._file_name("4.69800000000e-02") == file_name
        # 2D cylinder
        loader = TAUDataloader(join(self.path_1, self.file_name))
        file_name = join(self.path_0, "solution/solution") + \
            ".pval.unsteady_i=1000_t=6.0000e+02.domain_0"
        assert loader._file_name("6.0000e+02", ".domain_0")

    def test_load_domain_mesh_data(self):
        # backward facing step
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        data = loader._load_domain_mesh_data("0")
        assert data.shape == (13936 - 2118, 4)
        data = loader._load_domain_mesh_data("89")
        assert data.shape == (14472 - 2638, 4)
        # 2D cylinder
        loader = TAUDataloader(join(self.path_1, self.file_name), True)
        data = loader._load_domain_mesh_data("0")
        assert data.shape == (4510 - 173, 4)

    def test_load_mesh_data(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        n_points = 1119348
        assert loader.vertices.shape == (n_points, 3)
        assert loader.weights.shape == (n_points,)
        # backward facing step, distributed
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        assert loader.vertices.shape == (n_points, 3)
        assert loader.weights.shape == (n_points,)

    def test_write_times(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        assert loader.write_times[0] == "4.69800000000e-02"
        # backward facing step, distributed
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        assert loader.write_times[0] == "4.69800000000e-02"

    def test_field_names(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        times = loader.write_times
        assert "pressure" in loader.field_names[times[0]]
        assert "density" in loader.field_names[times[0]]
        # backward facing step, distributed
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        times = loader.write_times
        assert "pressure" in loader.field_names[times[0]]
        assert "density" in loader.field_names[times[0]]

    def test_load_snapshot(self):
        # backward facing step, serial
        loader = TAUDataloader(join(self.path_0, self.file_name))
        times = loader.write_times
        field_names = loader.field_names[times[-1]]
        n_points = 1119348
        # single snapshot, single field
        field = loader.load_snapshot(field_names[0], times[0])
        assert field.shape == (n_points,)
        # multiple snapshots, single field
        field_series = loader.load_snapshot(
            field_names[0], [times[0], times[0]])
        assert field_series.shape == (n_points, 2)
        assert pt.allclose(field_series[:, 0], field)
        # single snapshot, multiple fields
        f1, f2 = loader.load_snapshot(field_names[:2], times[0])
        assert f1.shape == (n_points,)
        assert f2.shape == (n_points,)
        # multiple snapshots, multiple field
        f1s,  f2s = loader.load_snapshot(field_names[:2], [times[0], times[0]])
        assert f1s.shape == (n_points, 2)
        assert f2s.shape == (n_points, 2)
        assert pt.allclose(f1s[:, 0], f1)
        assert pt.allclose(f2s[:, 0], f2)
        # backward facing step, distributed
        loader = TAUDataloader(join(self.path_0, self.file_name), True)
        # single snapshots, single field
        density = loader.load_snapshot("density", times[0])
        assert density.shape == (n_points,)
        # multiple snapshots, single field
        density = loader.load_snapshot("density", [times[0], times[0]])
        assert density.shape == (n_points, 2)
        # single snapshot, multiple fields
        density, p = loader.load_snapshot(["density", "pressure"], times[0])
        assert density.shape == (n_points, )
        assert p.shape == (n_points, )
        # multiple snapshots, multiple fields
        density, p = loader.load_snapshot(["density", "pressure"],
                                          [times[0], times[0]])
        assert density.shape == (n_points, 2)
        assert p.shape == (n_points, 2)
