"""Higher-order optimized DMD version.
"""

# standard library packages
from typing import Set
# third party packages
import torch as pt
# flowtorch packages
from .svd import SVD
from .optdmd import OptDMD
from .hodmd import _create_time_delays


class HOOptDMD(OptDMD):
    """Optimized DMD extended with data projection and time delay embedding.

    :param OptDMD: Optimized DMD base class
    :type OptDMD: `flowtorch.analysis.OptDMD`
    """
    def __init__(self, data_matrix: pt.Tensor, dt: float, delay: int = 1,
                 rank_dr: int = None, **dmd_options: dict):
        """Project data on POD basis and create time delay embedding.

        :param data_matrix: data matrix with snapshots organized as column vectors
        :type data_matrix: pt.Tensor
        :param dt: time step between two consecutive snapshots
        :type dt: float
        :param delay: number of time delay (history), defaults to 1 (no time delays)
        :type delay: int, optional
        :param rank_dr: number of POD modes to project the data on; an optimal value
            is determined by the `SVD` instance if None; defaults to None
        :type rank_dr: int, optional
        """
        self._dm_org = data_matrix
        self._rows_org, self._cols_org = data_matrix.shape
        self._svd_dr = SVD(data_matrix, rank_dr)
        super(HOOptDMD, self).__init__(
            _create_time_delays(self._svd_dr.U.T @ self._dm_org, delay),
            dt, **dmd_options
        )

    def partial_reconstruction(self, mode_indices: Set[int]) -> pt.Tensor:
        modes = self.modes
        mode_mask = pt.zeros(modes.shape[1], dtype=modes.dtype)
        mode_indices = pt.tensor(list(mode_indices), dtype=pt.int64)
        mode_mask[mode_indices] = 1.0
        rec = (modes * mode_mask) @ self.dynamics
        return rec.real.type(self._dmd._dm.dtype)

    @property
    def modes(self):
        modes = self.eigvecs / self.amplitude
        r = self._svd_dr.rank
        return self._svd_dr.U.type(modes.dtype) @ modes[:r]

    @property
    def reconstruction(self) -> pt.Tensor:
        return (self.modes @ self.dynamics).real

    @property
    def reconstruction_error(self) -> pt.Tensor:
        return self._dm_org[:, self._dmd._usecols] - self.reconstruction