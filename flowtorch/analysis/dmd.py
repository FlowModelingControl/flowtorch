"""Classes and functions to compute the dynamic mode decomposition (DMD) of a data matrix.
"""

# standard library packages
from typing import Tuple, Set, Union
# third party packages
import torch as pt
from numpy import pi
# flowtorch packages
from .svd import SVD
from flowtorch.data.utils import format_byte_size


def _dft_properties(dt: float, n_times: int) -> Tuple[float, float, float]:
    """Compute general properties of a discrete Fourier transformation.

    DFT properties like maximum frequency and frequency resolution can
    be a helpful guidance for building sensible data matrices used for
    modal decomposition.

    :param dt: timestep between two samples; assumed constant
    :type dt: float
    :param n_times: number of timesteps
    :type n_times: int
    :return: sampling frequency, maximum frequency, frequency resolution
    :rtype: Tuple[float, float, float]
    """
    fs = 1.0 / dt
    return fs, 0.5 * fs, fs / n_times



class DMD(object):
    """Class computing the exact DMD of a data matrix.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> from flowtorch.analysis import DMD
    >>> path = DATASETS["of_cavity_binary"]
    >>> loader = FOAMDataloader(path)
    >>> data_matrix = loader.load_snapshot("p", loader.write_times)
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3)
    >>> dmd.frequency
    tensor([0., 5., 0.])
    >>> dmd.growth_rate
    tensor([-2.3842e-06, -4.2345e+01, -1.8552e+01])
    >>> dmd.amplitude
    tensor([10.5635+0.j, -0.0616+0.j, -0.0537+0.j])
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3, robust=True)
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3, robust={"tol": 1.0e-5, "verbose" : True})

    """

    def __init__(self, data_matrix: pt.Tensor, dt: float, rank: int = None,
                 robust: Union[bool, dict] = False, unitary: bool = False,
                 optimal: bool = False, tlsq: bool = False, usecols: pt.Tensor = None):
        """Create DMD instance based on data matrix and time step. 

        :param data_matrix: data matrix whose columns are formed by the individual snapshots
        :type data_matrix: pt.Tensor
        :param dt: time step between two snapshots 
        :type dt: float
        :param rank: rank for SVD truncation, defaults to None
        :type rank: int, optional
        :param robust: data_matrix is split into low rank and sparse contributions
            if True or if dictionary with options for Inexact ALM algorithm; the SVD
            is computed only on the low rank matrix
        :type robust: Union[bool,dict]
        :param unitary: enforce the linear operator to be unitary; refer to piDMD_
            by Peter Baddoo for more information
        :type unitary: bool, optional
        :param optimal: compute mode amplitudes based on a least-squares problem
            as described in spDMD_ article by M. Janovic et al. (2014); in contrast
            to the original spDMD implementation, the exact DMD modes are used in
            the optimization problem as outlined in an article_ by R. Taylor
        :type optimal: bool, optional
        :param tlsq: de-biasing of the linear operator by solving a total least-squares
            problem instead of a standard least-squares problem; the rank is selected
            automatically or specified by the `rank` parameter; more information can be
            found in the TDMD_ article by M. Hemati et al.
        :type tlsq: bool, optional

        .. _piDMD: https://github.com/baddoo/piDMD
        .. _spDMD: https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        .. _article: http://www.pyrunner.com/weblog/2016/08/03/spdmd-python/
        .. _TDMD: http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf
        """
        self._dm = data_matrix
        self._complex = self._dm.dtype in (pt.complex32, pt.complex64, pt.complex128)
        self._rows, self._cols = self._dm.shape
        self._dt = dt
        self._unitary = unitary
        self._optimal = optimal
        self._tlsq = tlsq
        if self._tlsq:
            svd = SVD(pt.vstack((self._dm[:, :-1], self._dm[:, 1:])),
                      rank, robust)
            P = svd.V @ svd.V.conj().T
            self._X = self._dm[:, :-1] @ P
            self._Y = self._dm[:, 1:] @ P
            self._svd = SVD(self._X, svd.rank)
            del svd
        else:
            self._svd = SVD(self._dm[:, :-1], rank, robust)
            self._X = self._dm[:, :-1]
            self._Y = self._dm[:, 1:]
        self._eigvals, self._eigvecs, self._modes = self._compute_mode_decomposition()
        self._amplitude = self._compute_amplitudes()

    def _compute_operator(self):
        """Compute the approximate linear (DMD) operator.
        """
        if self._unitary:
            Xp = self._svd.U.conj().T @ self._X
            Yp = self._svd.U.conj().T @ self._Y
            U, _, VT = pt.linalg.svd(Yp @ Xp.conj().T, full_matrices=False)
            return U @ VT
        else:
            s_inv = pt.diag(1.0 / self._svd.s.type(self._dm.dtype))
            return self._svd.U.conj().T @ self._Y @ self._svd.V @ s_inv

    def _compute_mode_decomposition(self):
        """Compute reduced operator, eigen-decomposition, and DMD modes.
        """
        s_inv = pt.diag(1.0 / self._svd.s)
        operator = self._compute_operator()
        val, vec = pt.linalg.eig(operator)
        phi = (
            self._Y.type(val.dtype) @ self._svd.V.type(val.dtype)
            @ s_inv.type(val.dtype) @ vec
        )
        return val, vec, phi

    def _compute_amplitudes(self):
        """Compute amplitudes for exact DMD modes.

        If *optimal* is False, the amplitudes are computed based on the first
        snapshot in the data matrix; otherwise, a least-squares problem as
        introduced by Janovic et al. is solved (refer to the documentation
        in the constructor for more information).
        """
        if self._optimal:
            vander = pt.linalg.vander(self.eigvals, N=self._cols - 1)
            P = (self._modes.conj().T @ self._modes) * \
                (vander @ vander.conj().T).conj()
            q = pt.diag(vander @ self._X.type(P.dtype).conj().T @
                        self._modes).conj()
        else:
            P = self._modes
            q = self._dm[:, 0].type(P.dtype)
        return pt.linalg.lstsq(P, q).solution

    def partial_reconstruction(self, mode_indices: Set[int]) -> pt.Tensor:
        """Reconstruct data matrix with limited number of modes.

        :param mode_indices: mode indices to keep
        :type mode_indices: Set[int]
        :return: reconstructed data matrix
        :rtype: pt.Tensor
        """
        rows, cols = self._modes.shape
        mode_mask = pt.zeros(cols, dtype=pt.complex64)
        mode_indices = pt.tensor(list(mode_indices), dtype=pt.int64)
        mode_mask[mode_indices] = 1.0
        rec = (self.modes * mode_mask) @ self.dynamics
        if not self._complex:
            rec = rec.real
        return rec

    def top_modes(self, n: int = 10, integral: bool = False,
                  f_min: float = -float("inf"),
                  f_max: float = float("inf")) -> pt.Tensor:
        """Get the indices of the first n most important modes.

        Note that the conjugate complex modes for real data matrices are
        not filtered out by default. However, by setting the lower frequency
        threshold to a positive number, only modes with positive imaginary
        part are considered.

        :param n: number of indices to return; defaults to 10
        :type n: int
        :param integral: if True, the modes are sorted according to their
            integral contribution; defaults to False
        :type integral: bool, optional
        :param f_min: consider only modes with a frequency larger or equal
            to f_min; defaults to -inf
        :type f_min: float, optional
        :param f_max: consider only modes with a frequency smaller than f_max;
            defaults to -inf
        :type f_max: float, optional
        :return: indices of top n modes sorted by amplitude or integral
            contribution
        :rtype: pt.Tensor
        """
        importance = self.integral_contribution if integral else self.amplitude
        modes_in_range = pt.logical_and(self.frequency >= f_min,
                                        self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]),
                                 dtype=pt.int64)[modes_in_range]
        n = min(n, mode_indices.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]
    
    def predict(self, initial_condition: pt.Tensor, n_steps: int) -> pt.Tensor:
        """Predict evolution over N steps starting from used-defined initial conditions.

        :param initial_condition: initial state vector
        :type initial_condition: pt.Tensor
        :param n_steps: number of steps to predict
        :type n_steps: int
        :return: predicted evolution including the initial state (N+1 states are returned)
        :rtype: pt.Tensor
        """
        b = pt.linalg.pinv(self._modes) @ initial_condition.type(self._modes.dtype)
        prediction = self._modes @ pt.diag(b) @ pt.linalg.vander(self.eigvals, N=n_steps+1)
        if not self._complex:
            prediction = prediction.real
        return prediction

    @property
    def required_memory(self) -> int:
        """Compute the memory size in bytes of the DMD.

        :return: cumulative size of SVD, eigen values/vectors, and
            DMD modes in bytes
        :rtype: int
        """
        return (self._svd.required_memory +
                self._eigvals.element_size() * self._eigvals.nelement() +
                self._eigvecs.element_size() * self._eigvecs.nelement() +
                self._modes.element_size() * self._modes.nelement())

    @property
    def svd(self) -> SVD:
        return self._svd

    @property
    def operator(self) -> pt.Tensor:
        return self._compute_operator()

    @property
    def modes(self) -> pt.Tensor:
        return self._modes

    @property
    def eigvals(self) -> pt.Tensor:
        return self._eigvals
    
    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self._eigvals) / self._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        return self._eigvecs

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self._eigvals).imag / (2.0 * pi * self._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self._eigvals) / self._dt).real

    @property
    def amplitude(self) -> pt.Tensor:
        return self._amplitude

    @property
    def dynamics(self) -> pt.Tensor:
        return pt.diag(self.amplitude) @ pt.linalg.vander(self.eigvals, N=self._cols)

    @property
    def integral_contribution(self) -> pt.Tensor:
        """Integral contribution of individual modes according to J. Kou et al. 2017.

        DOI: https://doi.org/10.1016/j.euromechflu.2016.11.015
        """
        return self.modes.norm(dim=0)**2 * self.dynamics.abs().sum(dim=1)

    @property
    def reconstruction(self) -> pt.Tensor:
        """Reconstruct an approximation of the original data matrix.

        :return: reconstructed data matrix
        :rtype: pt.Tensor
        """
        rec = self.modes @ self.dynamics
        if not self._complex:
            rec = rec.real
        return rec

    @property
    def reconstruction_error(self) -> pt.Tensor:
        """Compute the reconstruction error.

        :return: difference between reconstruction and data matrix
        :rtype: pt.Tensor
        """
        return self.reconstruction - self._dm

    @property
    def projection_error(self) -> pt.Tensor:
        """Compute the difference between Y and AX.

        :return: projection error
        :rtype: pt.Tensor
        """
        YH = (self._modes @ pt.diag(self.eigvals)) @ \
            (pt.linalg.pinv(self._modes) @ self._X.type(self._modes.dtype))
        if self._complex:
            return YH - self._dm[:, 1:]
        else:
            return YH.real.type(self._Y.dtype) - self._dm[:, 1:]

    @property
    def tlsq_error(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the *noise* in X and Y.

        :return: noise in X and Y
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        if not self._tlsq:
            print("Warning: noise is only removed if tlsq=True")
        return self._dm[:, :-1] - self._X, self._dm[:, 1:] - self._Y
    
    @property
    def dft_properties(self) -> Tuple[float, float, float]:
        return _dft_properties(self._dt, self._cols - 1)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data_matrix, rank={self._svd.rank})"

    def __str__(self):
        ms = ["SVD:", str(self.svd), "LSQ:"]
        size, unit = format_byte_size(self.required_memory)
        ms.append("Overall DMD size: {:1.4f}{:s}".format(size, unit))
        ms.append("DFT frequencies (sampling, max., res.):")
        ms.append("{:1.4f}Hz, {:1.4f}Hz, {:1.4f}Hz".format(*self.dft_properties))
        ms.append("")
        return "\n".join(ms)
