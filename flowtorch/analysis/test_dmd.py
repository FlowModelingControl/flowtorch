# standard library packages
import pytest
# third party packages
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from flowtorch.analysis import DMD


def test_DMD():
    path = DATASETS["of_cavity_binary"]
    loader = FOAMDataloader(path)
    times = loader.write_times[1:]
    data = loader.load_snapshot("p", times)
    dmd = DMD(data, rank=3)
    assert dmd.modes.dtype == pt.complex64
    assert dmd.modes.shape[0] == data.shape[0]
    assert dmd.modes.shape[1] == 3
    growth, freq = dmd.spectrum(0.1)
    assert growth.shape[0] == 3
    assert freq.shape[0] == 3
    eigvals = dmd.eigvals
    assert eigvals.shape[0] == 3
    assert eigvals.dtype == pt.complex64
    eigvecs = dmd.eigvecs
    assert tuple(eigvecs.shape) == (3, 3)
    assert eigvecs.dtype == pt.complex64
