"""Classes and functions wrapping around *torch.linalg.svd*.
"""

# third party packages
import torch as pt
# flowtorch packages
from flowtorch.data.utils import format_byte_size


class SVD(object):
    def __init__(self, data_matrix: pt.Tensor, rank: int = None):
        shape = data_matrix.shape
        assert len(shape) == 2, (
            f"The data matrix must be a 2D tensor.\
            The provided data matrix has {len(shape)} dimensions."
        )
        self._rows, self._cols = shape
        U, s, VH = pt.linalg.svd(data_matrix, full_matrices=False)
        self._opt_rank = self._optimal_rank(s)
        if rank is None:
            self._rank = self.opt_rank
        else:
            self._rank = min(self._cols, rank)
        self._U = U[:, :self.rank]
        self._s = s[:self.rank]
        self._V = VH.conj().T[:, :self.rank]

    def _omega(self, beta: float):
        return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

    def _optimal_rank(self, s: pt.Tensor) -> int:
        assert len(s.shape) == 1, "Input must be a 1D tensor."
        beta = min(self._rows, self._cols) / max(self._rows, self._cols)
        tau_star = self._omega(beta) * pt.median(s)
        closest = pt.argmin((s - tau_star).abs()).item()
        if s[closest] > tau_star:
            return closest + 1
        else:
            return closest

    def required_memory(self) -> int:
        """Compute the memory size in bytes of the truncated SVD.

        :return: cumulative size of truncated U, s, and V tensors in bytes
        :rtype: int

        """
        return (self.U.element_size() * self.U.nelement() +
                self.s.element_size() * self.s.nelement() +
                self.V.element_size() * self.V.nelement())

    @property
    def U(self):
        return self._U

    @property
    def s(self):
        return self._s

    @property
    def V(self):
        return self._V

    @property
    def rank(self):
        return self._rank

    @property
    def opt_rank(self):
        return self._opt_rank

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data_matrix, rank={self.rank})"

    def __str__(self):
        ms = []
        ms.append(f"SVD of a {self._rows}x{self._cols} data matrix")
        ms.append(f"Selected/optimal rank: {self.rank}/{self.opt_rank}")
        ms.append(f"data type: {self.U.dtype} ({self.U.element_size()}b)")
        size, unit = format_byte_size(self.required_memory())
        ms.append("truncated SVD size: {:1.4f}{:s}".format(size, unit))
        return "\n".join(ms)
