# standard library packages
import pytest
# third party packages
import plotly.graph_objects as go
# flowtorch packages
from .psp_explorer import PSPExplorer

# this path will be replaced once non-proprietary test data is available
PATH = "/home/andre/Downloads/input/FOR/iPSP_reference_may_2021/0226.hdf5"

def test_psp_explorer():
    explorer = PSPExplorer(PATH)
    times = explorer.loader.write_times
    zones = explorer.loader.zone_names
    fig = explorer.interact(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)
    fig = explorer.mean(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)
    fig = explorer.std(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)