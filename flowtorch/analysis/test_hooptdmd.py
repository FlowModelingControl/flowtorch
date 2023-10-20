"""Unit tests for the higher-order OptDMD class.
"""

# third party packages
import torch as pt
# flowtorch packages
from .hooptdmd import HOOptDMD


class TestHOOptDMD():
    def test_init(self):
        dm = pt.rand((50, 15))
        dmd = HOOptDMD(dm, 1.0, rank_dr=10, rank=5)
        assert dmd.eigvals.size(0) == 5
        assert dmd.eigvecs.shape == (10, 5)

    def test_reconstruction(self):
        dm = pt.rand((50, 15))
        dmd = HOOptDMD(dm, 1.0, rank_dr=10, rank=5)
        modes = dmd.modes
        assert modes.shape == (50, 5)
        rec = dmd.reconstruction
        assert rec.shape == (50, 14)
        rec_err = dmd.reconstruction_error
        assert rec_err.shape == (50, 14)
        prec = dmd.partial_reconstruction({0, 2})
        assert prec.shape == (50, 14)
        