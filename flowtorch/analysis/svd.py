"""Classes and functions wrapping around *torch.linalg.svd*.
"""

# third party packages
import torch as pt
# flowtorch packages
from flowtorch.data.utils import format_byte_size


class SVD(object):
    """Compute and analyze the SVD of a data matrix.

    :param U: left singular vectors
    :type U: pt.Tensor
    :param s: singular values
    :type s: pt.Tensor
    :param s_rel: singular values normalized with their sum in percent
    :type s_rel: pt.Tensor
    :param s_cum: cumulative normalized singular values in percent
    :type s_cum: pt.Tensor
    :param V: right singular values
    :type V: pt.Tensor
    :param rank: rank used for truncation
    :type rank: int
    :param opt_rank: optimal rank according to SVHT
    :type opt_rank: int
    :param required_memory: memory required to store the truncation U, s, and V
    :type required_memory: int

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> from flowtorch.analysis import SVD
    >>> loader = FOAMDataloader(DATASETS["of_cavity_ascii"])
    >>> data = loader.load_snapshot("p", loader.write_times[1:])
    >>> svd = SVD(data, rank=100)
    >>> print(svd)
    SVD of a 400x5 data matrix
    Selected/optimal rank: 5/2
    data type: torch.float32 (4b)
    truncated SVD size: 7.9297Kb
    >>> svd.s_rel
    tensor([9.9969e+01, 3.0860e-02, 3.0581e-04, 7.8097e-05, 3.2241e-05])
    >>> svd.s_cum
    tensor([ 99.9687,  99.9996,  99.9999, 100.0000, 100.0000])
    >>> svd.U.shape
    torch.Size([400, 5])

    """

    def __init__(self, data_matrix: pt.Tensor, rank: int = None):
        shape = data_matrix.shape
        assert len(shape) == 2, (
            f"The data matrix must be a 2D tensor.\
            The provided data matrix has {len(shape)} dimensions."
        )
        self._rows, self._cols = shape
        U, s, VH = pt.linalg.svd(data_matrix, full_matrices=False)
        self._opt_rank = self._optimal_rank(s)
        self.rank = self.opt_rank if rank is None else rank
        self._U = U[:, :self.rank]
        self._s = s[:self.rank]
        self._V = VH.conj().T[:, :self.rank]

    def _optimal_rank(self, s: pt.Tensor) -> int:
        """Compute the optimal singular value hard threshold.

        This function implements the svht_ rank estimation.

        .. _svht: https://doi.org/10.1109/TIT.2014.2323359

        :param s: sorted singular values
        :type s: pt.Tensor
        :return: optimal rank for truncation
        :rtype: int
        """
        beta = min(self._rows, self._cols) / max(self._rows, self._cols)
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        tau_star = omega * pt.median(s)
        closest = pt.argmin((s - tau_star).abs()).item()
        if s[closest] > tau_star:
            return closest + 1
        else:
            return closest

    def reconstruct(self, rank: int = None) -> pt.Tensor:
        """Reconstruct the data matrix for a given rank.

        :param rank: rank used to compute a truncated reconstruction
        :type rank: int, optional
        :return: reconstruction of the input data matrix
        :rtype: pt.Tensor
        """
        r_rank = self.rank if rank is None else max(min(rank, self.rank), 1)
        return self.U[:, :r_rank] @ pt.diag(self.s[:r_rank]) @ self.V[:, :r_rank].conj().T

    @property
    def U(self) -> pt.Tensor:
        return self._U

    @property
    def s(self) -> pt.Tensor:
        return self._s

    @property
    def s_rel(self) -> pt.Tensor:
        return self._s / self._s.sum() * 100.0

    @property
    def s_cum(self) -> pt.Tensor:
        s_sum = self._s.sum().item()
        return pt.tensor(
            [self._s[:i].sum().item() / s_sum * 100.0 for i in range(1, self._s.shape[0]+1)],
            dtype=self._s.dtype
        )

    @property
    def V(self) -> pt.Tensor:
        return self._V

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, value: int):
        self._rank = max(min(self._cols, value), 1)
        self._cols = self._rank

    @property
    def opt_rank(self) -> int:
        return self._opt_rank

    @property
    def required_memory(self) -> int:
        """Compute the memory size in bytes of the truncated SVD.

        :return: cumulative size of truncated U, s, and V tensors in bytes
        :rtype: int
        """
        return (self.U.element_size() * self.U.nelement() +
                self.s.element_size() * self.s.nelement() +
                self.V.element_size() * self.V.nelement())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(data_matrix, rank={self.rank})"

    def __str__(self) -> str:
        ms = []
        ms.append(f"SVD of a {self._rows}x{self._cols} data matrix")
        ms.append(f"Selected/optimal rank: {self.rank}/{self.opt_rank}")
        ms.append(f"data type: {self.U.dtype} ({self.U.element_size()}b)")
        size, unit = format_byte_size(self.required_memory)
        ms.append("truncated SVD size: {:1.4f}{:s}".format(size, unit))
        return "\n".join(ms)
