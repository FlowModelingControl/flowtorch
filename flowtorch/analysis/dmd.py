"""Classes and functions to compute the dynamic mode decomposition (DMD) of a data matrix.
"""

# standard library packages
from typing import Tuple
# third party packages
import torch as pt
from numpy import pi
# flowtorch packages
from .svd import SVD
from flowtorch.data.utils import format_byte_size


class DMD(object):
    def __init__(self, data_matrix: pt.Tensor, rank: int = None):
        self._dm = data_matrix
        self._svd = SVD(self._dm[:, :-1], rank)
        self._eigvals, self._eigvecs, self._modes = self._compute_mode_decomposition()

    def _compute_mode_decomposition(self):
        """Compute reduced operator, eigen decomposition, and DMD modes.
        """
        s_inv = pt.diag(1.0 / self._svd.s)
        operator = (
            self._svd.U.conj().T @ self._dm[:, 1:] @ self._svd.V @ s_inv
        )
        val, vec = pt.linalg.eig(operator)
        # type conversion is currently not implemented for pt.complex32
        # such that the dtype for the modes is always pt.complex64
        phi = (
            self._dm[:, 1:].type(val.dtype) @ self._svd.V.type(val.dtype)
            @ s_inv.type(val.dtype) @ vec
        )
        return val, vec, phi

    def spectrum(self, dt: float) -> Tuple[pt.Tensor, pt.Tensor]:
        omega = pt.log(self._eigvals) / dt
        return omega.real, omega.imag / (2.0 * pi)

    def required_memory(self) -> int:
        """Compute the memory size in bytes of the DMD.

        :return: cumulative size of SVD, eigen values/vectors, and
            DMD modes in bytes
        :rtype: int

        """
        return (self._svd.required_memory() +
                self._eigvals.element_size() * self._eigvals.nelement() +
                self._eigvecs.element_size() * self._eigvecs.nelement() +
                self._modes.element_size() * self._modes.nelement())

    @property
    def svd(self):
        return self._svd

    @property
    def modes(self):
        return self._modes

    @property
    def eigvals(self):
        return self._eigvals

    @property
    def eigvecs(self):
        return self._eigvecs

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data_matrix, rank={self._svd.rank})"

    def __str__(self):
        ms = ["SVD:", str(self.svd), "LSQ:"]
        size, unit = format_byte_size(self.required_memory())
        ms.append("Overall DMD size: {:1.4f}{:s}".format(size, unit))
        return "\n".join(ms)