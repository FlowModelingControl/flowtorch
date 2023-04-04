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


class DMD(object):
    """Class computing the exact DMD of a data matrix.

    Currently, no advanced mode selection algorithms are implemented.
    The mode amplitudes are computed using the first snapshot.

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
        :param usecols: tensor of column indices to build data matrix X and shifted
            data matrix Y; used to implement bagging and ensemble DMD; if not specified,
            X and Y are based on all but the last and all but the first columns, respectively;
            the column indices are expected to be in ascending order
        :type usecols: pt.Tensor, optional


        .. _piDMD: https://github.com/baddoo/piDMD
        .. _spDMD: https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        .. _article: http://www.pyrunner.com/weblog/2016/08/03/spdmd-python/
        .. _TDMD: http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf
        """
        self._dm = data_matrix
        self._rows, self._cols = self._dm.shape
        self._dt = dt
        self._unitary = unitary
        self._optimal = optimal
        self._tlsq = tlsq
        self._usecols = usecols
        if self._usecols is None:
            self._usecols = pt.tensor(range(self._cols - 1), dtype=pt.int64)
        self._validate_inputs()
        if self._tlsq:
            svd = SVD(pt.vstack((self._dm[:, self._usecols], self._dm[:, self._usecols + 1])),
                      rank, robust)
            P = svd.V @ svd.V.conj().T
            self._X = self._dm[:, self._usecols] @ P
            self._Y = self._dm[:, self._usecols + 1] @ P
            self._svd = SVD(self._X, svd.rank)
            del svd
        else:
            self._svd = SVD(self._dm[:, self._usecols], rank, robust)
            self._X = self._dm[:, self._usecols]
            self._Y = self._dm[:, self._usecols + 1]
        self._eigvals, self._eigvecs, self._modes = self._compute_mode_decomposition()
        self._amplitude = self._compute_amplitudes()

    def _validate_inputs(self):
        """Validate input values.

        :raises ValueError: if more indices than allowed are passed via usecols; the maximum
            number of indices is one less than the number of snapshots (we still need to build a
            shift matrix)
        :raises ValueError: if usecols contains the index of the last snapshot; the last state
            has no corresponding state shifted by one time step
        """
        if len(self._usecols) >= self._cols:
            raise ValueError(f"Parameter usecols contains too many indices:\n" + 
                             f"{len(self._usecols):d} (maximum {self._cols - 1:d})"
            )
        if self._cols - 1 in self._usecols:
            raise ValueError(
                "The parameter usecols must not contain the index of the last column; " + 
                "otherwise, no shifted data matrix can be built"
            )

    def _compute_operator(self):
        """Compute the approximate linear (DMD) operator.
        """
        if self._unitary:
            Xp = self._svd.U.conj().T @ self._X
            Yp = self._svd.U.conj().T @ self._Y
            U, _, VT = pt.linalg.svd(Yp @ Xp.conj().T, full_matrices=False)
            return U @ VT
        else:
            s_inv = pt.diag(1.0 / self._svd.s)
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
    
    def _compute_vander_mode_matrix(self):
        """Compute the Vandermode matrix.

        The Vandermode matrix is useful to evaluate the DMD prediction in
        one shot/one matrix multiplication. This option is useful to
        compute optimized amplitudes or to evaluate the modes' dynamics.
        If the full data matrix is used, the Vandermode matrix is easily
        computed using torch.vander; however, if only selected columns
        are used to fit the DMD operator, the corresponding matrix must
        be built manually (by looping).        
        """
        if len(self._usecols) == self._cols - 1:
            return pt.vander(self.eigvals, self._cols - 1, True)
        else:
            exponents = self._usecols - self._usecols.min()
            vander = pt.zeros(
                (self.eigvals.shape[0], exponents.shape[0]), dtype=self.eigvals.dtype
            )
            for i, e in enumerate(exponents):
                vander[:, i] = self.eigvals**e
            return vander

    def _compute_amplitudes(self):
        """Compute amplitudes for exact DMD modes.

        If *optimal* is False, the amplitudes are computed based on the first
        snapshot in the data matrix; otherwise, a least-squares problem as
        introduced by Janovic et al. is solved (refer to the documentation
        in the constructor for more information).
        """
        if self._optimal:
            vander = self._compute_vander_mode_matrix()
            P = (self._modes.conj().T @ self._modes) * \
                (vander @ vander.conj().T).conj()
            q = pt.diag(vander @ self._X.type(P.dtype).conj().T @
                        self._modes).conj()
        else:
            P = self._modes
            q = self._X[:, self._usecols[0]].type(P.dtype)
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
        reconstruction = (self.modes * mode_mask) @ self.dynamics
        if self._dm.dtype in (pt.complex128, pt.complex64, pt.complex32):
            return reconstruction.type(self._dm.dtype)
        else:
            return reconstruction.real.type(self._dm.dtype)

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
        n = min(n, modes_in_range.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]

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
        return pt.diag(self.amplitude) @ self._compute_vander_mode_matrix()

    @property
    def integral_contribution(self) -> pt.Tensor:
        """Integral contribution of individual modes according to J. Kou et al. 2017.

        DOI: https://doi.org/10.1016/j.euromechflu.2016.11.015
        """
        return self.modes.norm(dim=0)**2 * self.dynamics.abs().sum(dim=1)

    @property
    def reconstruction(self) -> pt.Tensor:
        """Reconstruct an approximation of the training data.

        :return: reconstructed training data
        :rtype: pt.Tensor
        """
        if self._dm.dtype in (pt.complex128, pt.complex64, pt.complex32):
            return (self.modes @ self.dynamics).type(self._dm.dtype)
        else:
            return (self.modes @ self.dynamics).real.type(self._dm.dtype)

    @property
    def reconstruction_error(self) -> pt.Tensor:
        """Compute the reconstruction error.

        :return: difference between reconstruction and data matrix
        :rtype: pt.Tensor
        """
        return self.reconstruction - self._dm[:, self._usecols]

    @property
    def projection_error(self) -> pt.Tensor:
        """Compute the difference between Y and AX.

        :return: projection error
        :rtype: pt.Tensor
        """
        YH = (self._modes @ pt.diag(self.eigvals)) @ \
            (pt.linalg.pinv(self._modes) @ self._X.type(self._modes.dtype))
        if self._Y.dtype in (pt.complex128, pt.complex64, pt.complex32):
            return YH - self._dm[:, self._usecols + 1]
        else:
            return YH.real.type(self._Y.dtype) - self._dm[:, self._usecols + 1]

    @property
    def tlsq_error(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the *noise* in X and Y.

        :return: noise in X and Y
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        if not self._tlsq:
            print("Warning: noise is only removed if tlsq=True")
        return self._dm[:, self._usecols] - self._X, self._dm[:, self._usecols + 1] - self._Y

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data_matrix, rank={self._svd.rank})"

    def __str__(self):
        ms = ["SVD:", str(self.svd), "LSQ:"]
        size, unit = format_byte_size(self.required_memory)
        ms.append("Overall DMD size: {:1.4f}{:s}".format(size, unit))
        return "\n".join(ms)
