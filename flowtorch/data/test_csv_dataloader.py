# standard library packages
import pytest
# third party packages
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import CSVDataloader
from flowtorch.data.csv_dataloader import _parse_davis_header


def test_from_foam_surface():
    path = DATASETS["csv_naca0012_alpha4_surface"]
    loader = CSVDataloader.from_foam_surface(
        path, "total(p)_coeff_airfoil.raw", "cp")
    times = loader.write_times
    assert len(times) == 250
    assert times[0] == "0.001"
    assert times[-1] == "0.25"
    fields = loader.field_names
    assert fields[times[0]][0] == "cp"
    n_points = 28892
    vertices = loader.vertices
    assert vertices.shape == (n_points, 3)
    weights = loader.weights
    # should be a tensor filled with ones
    assert pt.sum(weights).item() == n_points
    # single snapshot, single field
    snapshot = loader.load_snapshot("cp", times[0])
    assert snapshot.shape == (n_points,)
    # multiple snapshots, single field
    snapshots = loader.load_snapshot("cp", times[:10])
    assert snapshots.shape == (n_points, 10)
    assert pt.allclose(snapshot, snapshots[:, 0])
    # single snapshot, multiple fields
    snapshot = loader.load_snapshot(["cp"], times[0])
    assert len(snapshot) == 1
    assert snapshot[0].shape == (n_points,)
    # multiple snapshots, multiple fields
    snapshots = loader.load_snapshot(["cp"], times[:10])
    assert len(snapshots) == 1
    assert snapshots[0].shape == (n_points, 10)
    assert pt.allclose(snapshots[0][:, 0], snapshot[0])


def test_parse_davis_header():
    header = 'VARIABLES = "x", "y", "Vx", "Vy", "Vz", "swirl strength", "vector length", "vorticity", "isValid"'
    columns = _parse_davis_header(header)
    expected_columns = ["x", "y", "Vx", "Vy", "Vz", "swirl strength", "vector length", "vorticity", "isValid"]
    assert columns == expected_columns


def test_from_davis():
    path = DATASETS["csv_aoa8_beta0_xc100_stereopiv"]
    loader = CSVDataloader.from_davis(path, "B")
    times = loader.write_times
    assert len(times) == 5000
    assert times[0] == "00001"
    assert times[-1] == "05000"
    fields = loader.field_names
    fields = fields[times[0]]
    n_points = 3741
    vertices = loader.vertices
    # DaVis files have only x and y component
    assert vertices.shape == (n_points, 2)
    weights = loader.weights
    assert weights.shape == (n_points,)
    # single snapshot, single field
    snapshot = loader.load_snapshot("Vx", times[0])
    assert snapshot.shape == (n_points,)
    # multiple snapshots, single field
    snapshots = loader.load_snapshot("Vx", times[:10])
    assert snapshots.shape == (n_points, 10)
    assert pt.allclose(snapshot, snapshots[:, 0])
    # single snapshot, multiple fields
    snapshot = loader.load_snapshot(fields[:3], times[0])
    assert len(snapshot) == 3
    assert snapshot[0].shape == (n_points,)
    assert snapshot[1].shape == (n_points,)
    assert snapshot[2].shape == (n_points,)
    # multiple snapshots, multiple fields
    snapshots = loader.load_snapshot(fields[:3], times[:10])
    assert len(snapshots) == 3
    assert snapshots[0].shape == (n_points, 10)
    assert snapshots[1].shape == (n_points, 10)
    assert snapshots[2].shape == (n_points, 10)
    assert pt.allclose(snapshots[0][:, 0], snapshot[0])
    # test parsing multiple fields (header containing more than velocity)
    path = DATASETS["csv_davis_multiple_fields"]
    loader = CSVDataloader.from_davis(path, "B")
    times = loader.write_times
    fields = loader.field_names[times[0]]
    assert fields == ["Vx", "Vy", "Vz", "swirl strength", "vector length", "vorticity"]
    snapshot = loader.load_snapshot("swirl strength", times[0])
    assert snapshot.shape == (n_points,)
