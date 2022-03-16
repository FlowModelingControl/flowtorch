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
    rank = 3
    dmd = DMD(data, dt=0.1, rank=rank)
    assert dmd.eigvals.shape == (rank,)
    assert dmd.eigvals.dtype == pt.complex64
    assert dmd.eigvecs.shape == (rank, rank)
    assert dmd.eigvecs.dtype == pt.complex64
    assert dmd.modes.dtype == pt.complex64
    assert dmd.modes.shape == (data.shape[0], rank)
    assert dmd.frequency.shape == (rank,)
    assert dmd.growth_rate.shape == (rank,)
    assert dmd.amplitude.shape == (rank,)
    assert dmd.amplitude.dtype == pt.complex64
    assert dmd.dynamics.shape == (rank, data.shape[-1])
    assert dmd.dynamics.dtype == pt.complex64
    assert dmd.integral_contribution.shape == (rank,)
    assert dmd.integral_contribution.dtype == pt.float32
    assert dmd.reconstruction.shape == data.shape
    assert dmd.reconstruction.dtype == data.dtype
    partial = dmd.partial_reconstruction({0})
    assert partial.dtype == data.dtype
    assert partial.shape == data.shape
    partial = dmd.partial_reconstruction({0, 2})
    assert partial.dtype == data.dtype
    assert partial.shape == data.shape
    top = dmd.top_modes(10)
    top = dmd.top_modes(10, True)
    assert top.shape == (min(rank, 10),)
    assert top.dtype == pt.int64
    # robust DMD
    dmd = DMD(data, dt=0.1, rank=rank, robust=True)
    assert dmd.svd.L.shape == (data.shape[0], rank+1)
    # unitary operator
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True)
    operator = dmd.operator
    shape = operator.shape
    assert shape == (rank, rank)
    diag = operator.conj().T @ operator
    assert pt.allclose(diag, pt.diag(pt.ones(rank)), atol=1e-6)
    # optimal mode amplitudes
    dmd = DMD(data, dt=0.1, rank=rank, optimal=True)
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True, optimal=True)
    assert dmd.amplitude.shape == (rank,)
    assert dmd.amplitude.dtype == pt.complex64




