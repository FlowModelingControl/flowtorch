"""Unit tests for the OptDMD class.
"""

# standard library packages
from os import remove
from os.path import join, isfile
# third party packages
import torch as pt
# flowtorch packages
from .optdmd import EarlyStopping, OptDMD


class TestEarlyStopping():
    def test_init(self):
        stopper = EarlyStopping()
        stop = stopper(1.0)
        assert not stop
        assert stopper._best_loss == 1.0
        _ = stopper(2.0)
        assert stopper._counter == 1

    def test_checkpoint(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, 5)
        chp = join("/tmp", "optDMDchp.pt")
        stopper = EarlyStopping(checkpoint=chp, model=dmd)
        _ = stopper(1.0)
        assert isfile(chp)
        eigs_before = dmd.eigvals
        dmd.train(3)
        dmd.load_state_dict(pt.load(chp))
        eigs_after = dmd.eigvals
        assert all(pt.isclose(eigs_after, eigs_before))
        if isfile(chp):
            remove(chp)

class TestOptDMD():
    def test_init(self):
        dm = pt.rand((50, 20))
        # construction with even rank
        rank = 10
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        # construction with odd rank
        rank = 11
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        # construction with complex data matrix
        dm = pt.rand((50, 20), dtype=pt.complex64)
        rank = 11
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        assert dmd.modes.shape == (50, rank)

    def test_create_train_val_split(self):
        dm = pt.rand((50, 21))
        dmd = OptDMD(dm, 1.0, 10)
        train, val = dmd._create_train_val_split(0.5, 0.5)
        assert len(train) == 10
        assert len(val) == 10
        assert train[:][0].tolist() == list(range(10))
        assert val[:][0].tolist() == list(range(10, 20))

    def test_forward(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, 10)
        test_input = (pt.tensor(range(5), dtype=pt.int64),)
        pred = dmd(test_input)
        assert pred.shape == (50, 5)
        assert pred.requires_grad

    def test_train(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=10)
        dmd.train(5)
        assert len(dmd.log["train_loss"]) == 5
        assert isinstance(dmd.log["train_loss"][-1], float)
        assert len(dmd.log["val_loss"]) == 5
        assert isinstance(dmd.log["val_loss"][-1], float)
        # no validation data
        dmd = OptDMD(dm, 1.0, 10)
        dmd.train(5, val_size=0, loss_key="train_loss")
        assert "val_loss" not in dmd.log
        # batch size larger than possible
        dmd.train(5, batch_size=200)
        # train with complex data matrix
        dm = pt.rand((50, 20), dtype=pt.complex64)
        dmd = OptDMD(dm, 1.0, rank=10)
        dmd.train(5)
        assert True

    def test_partial_reconstruction(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=10)
        rec = dmd.partial_reconstruction((0, 1))
        assert rec.shape == (50, 19)
        assert rec.dtype == dm.dtype

    def test_top_modes(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=5)
        top = dmd.top_modes(1, True, 0.0)
        assert len(top) == 1
        im = dmd.integral_contribution
        assert im[top] == im.max()

    def test_reconstruction(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=5)
        rec = dmd.reconstruction
        assert rec.shape == (50, 19)
        assert rec.dtype == dm.dtype
