"""Classes and functions to compute the dynamic mode decomposition (DMD) of a data matrix.
"""

# standard library packages
from typing import Tuple
# third party packages
import torch as pt
import numpy as np
# flowtorch packages
from .svd import SVD
from flowtorch.data.utils import format_byte_size


DTYPE_MAPPING = {
    pt.float32: pt.complex32,
    pt.float64: pt.complex64,
    pt.complex32: pt.complex32,
    pt.complex64: pt.complex64
}


class DMD(object):
    def __init__(self, data_matrix: pt.Tensor, rank: int = None):
        self._dm = data_matrix
        self._svd = SVD(self._dm[:, :-1], rank)
        self._eigvals, self._eigvecs, self._modes = self._compute_mode_decomposition()

    def _compute_mode_decomposition(self):
        """Compute DMD modes and corresponding eigenvalues.

        .. note::
        This function will be changed with the release of PyTorch 1.9,
        which introduces CPU/GPU enabled eigen-decomposition for complex
        matrices. Until then, `numpy.linalg.eig` will be used.

        """
        s_inv = pt.diag(1.0 / self._svd.s)
        operator = (
            self._svd.U.conj().T @ self._dm[:, 1:] @ self._svd.V @ s_inv
        )
        val, vec = np.linalg.eig(operator.numpy())
        # type conversion is currently not implemented for pt.complex32
        # for now, the dtype is always set to pt.complex64
        # dtype = DTYPE_MAPPING.get(self._dm.dtype, pt.complex64)
        dtype = pt.complex64
        val = pt.from_numpy(val).type(dtype)
        vec = pt.from_numpy(vec).type(dtype)
        phi = (
            self._dm[:, 1:].type(dtype) @ self._svd.V.type(dtype)
            @ s_inv.type(dtype) @ vec
        )
        return val, vec, phi

    def spectrum(self, dt: float) -> Tuple[pt.Tensor, pt.Tensor]:
        omega = pt.log(self._eigvals) / dt
        return omega.real, omega.imag / (2.0 * np.pi)

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