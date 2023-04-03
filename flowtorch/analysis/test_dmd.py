"""Unit tests for the DMD class.
"""

# third party packages
from pytest import raises
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
    rows, cols = data.shape
    rank = 3
    dmd = DMD(data, dt=0.1, rank=rank)
    assert dmd.eigvals.shape == (rank,)
    assert dmd.eigvals.dtype == pt.complex64
    assert dmd.eigvecs.shape == (rank, rank)
    assert dmd.eigvecs.dtype == pt.complex64
    assert dmd.modes.dtype == pt.complex64
    assert dmd.modes.shape == (rows, rank)
    assert dmd.frequency.shape == (rank,)
    assert dmd.growth_rate.shape == (rank,)
    assert dmd.amplitude.shape == (rank,)
    assert dmd.amplitude.dtype == pt.complex64
    assert dmd.dynamics.shape == (rank, cols - 1)
    assert dmd.dynamics.dtype == pt.complex64
    assert dmd.integral_contribution.shape == (rank,)
    assert dmd.integral_contribution.dtype == pt.float32
    assert dmd.reconstruction.shape == (rows, cols - 1)
    assert dmd.reconstruction.dtype == data.dtype
    partial = dmd.partial_reconstruction({0})
    assert partial.dtype == data.dtype
    assert partial.shape == (rows, cols - 1)
    partial = dmd.partial_reconstruction({0, 2})
    assert partial.dtype == data.dtype
    assert partial.shape == (rows, cols - 1)
    top = dmd.top_modes(10)
    top = dmd.top_modes(10, True)
    assert top.shape == (min(rank, 10),)
    assert top.dtype == pt.int64
    assert dmd.reconstruction_error.shape == (rows, cols - 1)
    assert dmd.projection_error.shape == (rows, cols - 1)
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
    # total least-squares
    dmd = DMD(data, dt=0.1, tlsq=True)
    assert dmd.amplitude.dtype == pt.complex64
    dmd = DMD(data, dt=0.1, rank=rank, optimal=True, tlsq=True)
    assert dmd.amplitude.shape == (rank,)
    DX, DY = dmd.tlsq_error
    assert  DX.shape == (data.shape[0], data.shape[1] - 1)
    assert  DY.shape == (data.shape[0], data.shape[1] - 1)
    # DMD with usecols parameter
    with raises(ValueError):
        usecols = pt.tensor([0, data.shape[1] - 1], dtype=pt.int64)
        dmd = DMD(data, dt=0.1, usecols=usecols)
    with raises(ValueError):
        usecols = pt.ones(data.shape[1], dtype=pt.int64)
        dmd = DMD(data, dt=0.1, usecols=usecols)
    dmd = DMD(data, dt=0.1, rank=3, optimal=True, usecols=pt.tensor([0, 2, 3], dtype=pt.int64))
    rec = dmd.reconstruction
    assert rec.shape == (rows, 3)






