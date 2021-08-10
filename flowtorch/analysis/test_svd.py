# third party libraries
import pytest
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from flowtorch.analysis import SVD

class TestSVD():
    def setup_method(self, test_method):
        loader = FOAMDataloader(DATASETS["of_cavity_ascii"])
        self.data = loader.load_snapshot("p", loader.write_times[1:])
        self.rows, self.cols = self.data.shape

    def test_init(self):
        svd = SVD(self.data, rank=self.cols)
        assert svd.U.shape == self.data.shape
        assert svd.V.shape == (self.cols, self.cols)
        assert svd.s.shape == (self.cols,)
        assert svd.s_rel.shape == (self.cols,)
        assert svd.s_cum.shape == (self.cols,)
        assert pt.allclose(svd.s_cum[-1], pt.tensor(100.0))

    def test_optimal_rank(self):
        svd = SVD(self.data)
        assert 1 <= svd.opt_rank <= self.cols
        assert svd._cols == svd.rank

    def test_reconstruct(self):
        data = self.data.type(pt.float64)
        svd = SVD(data, self.cols*2)
        assert pt.allclose(data, svd.reconstruct())
        assert pt.allclose(data, svd.reconstruct(self.cols*2))
        err_r1 = pt.linalg.norm(data - svd.reconstruct(rank=1)).item()
        err_r2 = pt.linalg.norm(data - svd.reconstruct(rank=2)).item()
        assert err_r2 <= err_r1