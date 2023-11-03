"""Optimized DMD via gradient descent and backpropagation.
"""

# standard library packages
from typing import Union, Set, Callable, Tuple
from collections import defaultdict
from math import sqrt

# third party packages
import torch as pt
from numpy import pi

# flowtorch packages
from .dmd import DMD


DEFAULT_SCHEDULER_OPT = {"mode": "min", "factor": 0.5, "patience": 20, "min_lr": 1.0e-6}


def _create_conj_complex_pairs(ev: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    """Create indices to preserve conjugate complex eigenvalues.

    For real input data, eivenvalues and eigenvectors must come as
    conjugate complex pairs. If there is on odd number of eigenvalues,
    one eigenvalue is reserved to be on the real axis. If there are
    multiple eigenvalues without imaginary part and the number of eigenvalues
    is odd, only the one with real part closest to unity is preserved.

    :param ev: tensor of eigenvalues
    :type ev: pt.Tensor
    :return: indices of eigenvalues to keep and indices of eigenvalues
         with conjugate complex pair
    :rtype: Tuple[pt.Tensor, pt.Tensor]
    """
    sort_imag = ev.imag.sort(descending=True).indices
    zero_imag = ev.imag == 0
    n = ev.size(0)
    indices = pt.tensor(range(n), dtype=pt.int64)
    pairs = indices[: n // 2]
    if zero_imag.sum() <= 1:  # unique real ev
        keep = indices[sort_imag][: n // 2 + n % 2]
    else:  # multiple real ev
        if n % 2 == 1:  # preserve ev with real part closest to unity
            keep_real = (ev[zero_imag].real - 1.0).abs().sort().indices[0]
            keep_real = indices[zero_imag][keep_real]
            keep = pt.cat((indices[sort_imag][: n // 2], keep_real.unsqueeze(-1)))
        else:  # do not preserve any real ev
            keep = indices[sort_imag][: n // 2]
    return keep, pairs


def l2_loss(
    label: pt.Tensor, prediction: pt.Tensor, eigvecs: pt.Tensor, eigvals: pt.Tensor
) -> pt.Tensor:
    """Compute the L2 norm of the prediction error.

    Note: in contrast to the default norm function in PyTorch,
    the norm is normalized by the number of elements in the
    prediction tensor.

    Note: the argument list is more generic than necessary for this
    loss function; the eigenvectors and eigenvalues may be used for
    regularization or to enforce physical constraints in more advanced
    loss functions.

    :param label: ground truth
    :type label: pt.Tensor
    :param prediction: predictions
    :type prediction: pt.Tensor
    :return: L2 norm normalized by number of elements
    :rtype: pt.Tensor
    """
    return (label - prediction).norm() / sqrt(prediction.numel())


class EarlyStopping:
    """Provide stopping control for iterative optimization tasks."""

    def __init__(
        self,
        patience: int = 40,
        min_delta: float = 0.0,
        checkpoint: str = None,
        model: pt.nn.Module = None,
    ):
        """Initialize a new controller instance.

        :param patience: number of iterations to wait for an improved
            loss value before stopping; defaults to 40
        :type patience: int, optional
        :param min_delta: minimum reduction in the loss values to be
            considered an improvement; avoids overly long optimization
            with marginal improvements per iteration; defaults to 0.0
        :type min_delta: float, optional
        :param checkpoint: path at which to store the best known state
            of the model; the state is not saved if None; defaults to None
        :type checkpoint: str, optional
        :param model: instance of PyTorch model; the model's state dict is
            saved upon improvement of the loss function is a valid checkpoint
            is provided; defaults to None
        :type model: pt.nn.Module, optional
        """
        self._patience = patience
        self._min_delta = min_delta
        self._chp = checkpoint
        self._model = model
        self._best_loss = float("inf")
        self._counter = 0
        self._stop = False

    def __call__(self, loss: float) -> bool:
        """_summary_

        :param loss: new loss value
        :type loss: float
        :return: boolean flag indicating if the optimization can be stopped
        :rtype: bool
        """
        if loss < self._best_loss - self._min_delta:
            self._best_loss = loss
            self._counter = 0
            if self._chp is not None and self._model is not None:
                pt.save(self._model.state_dict(), self._chp)
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stop = True
        return self._stop


class OptDMD(pt.nn.Module):
    """Optimized DMD based on backpropagation and gradient descent.

    Examples

    >>> import torch as pt
    >>> from flowtorch.analysis import OptDMD
    >>> dm = pt.rand((200, 100))
    >>> dmd = DMD(dm, dt=1.0)
    >>> dmd.train(stopping_options={'patience' : 80, 'checkpoint' : '/tmp/best_model.pt'})
    >>> dmd.load_state_dict(pt.load('/tmp/best_model.pt'))
    >>> train_loss = dmd.log['train_loss']
    >>> val_loss = dmd.log['val_loss']
    """

    def __init__(self, *dmd_args, **dmd_kwargs):
        """Construct an optimized DMD instance.

        The arguments and keyword arguments passed to the constructor are
        used to create a regular `DMD` instance for initialization. The
        arguments specific to the optimization of eigenvectors and eigenvalues
        are passed to the `train` method.

        Warning: the implementation was only tested rigorously for real input
        data; eigenvectors and eigenvalues are enforced to have complex conjugate
        pairs in case of real input data. If there is an odd number of modes,
        the mode-eigenvalue-pair whose imaginary part
        """
        super(OptDMD, self).__init__()
        self._dmd = DMD(*dmd_args, **dmd_kwargs)
        if self._dmd._complex:
            n_modes = self._dmd.eigvals.size(0)
            keep = pt.tensor(range(n_modes), dtype=pt.int64)
            self._conj_indices = pt.zeros(n_modes, dtype=pt.bool)
        else:
            keep, pairs = _create_conj_complex_pairs(self._dmd.eigvals)
            self._conj_indices = pairs
        scaled_modes = self._dmd._modes[:, keep] * self._dmd.amplitude[keep]
        self._eigvecs = pt.nn.Parameter(scaled_modes)
        self._eigvals = pt.nn.Parameter(self._dmd._eigvals[keep].clone())
        self._log = defaultdict(list)

    def _create_train_val_split(
        self, train_size: Union[int, float], val_size: Union[int, float]
    ) -> tuple:
        """Split the data matrix into training and validation data.

        In contrast to general machine learning tasks, the data is not
        split randomly. The validation data are taken from the end of
        the snapshot sequence. The provided values for training and validation
        size are weighted by their sum such that always a valid split results.

        :param train_size: size of the training data set; can be the number
            of the snapshots or a fraction of the overall number of snapshots
        :type train_size: Union[int, float]
        :param val_size: size of the validation data set; can be the number
            of the snapshots or a fraction of the overall number of snapshots
        :type val_size: Union[int, float]
        :return: training and validation data wrapped as `TensorDataset`
        :rtype: tuple
        """
        data = pt.utils.data.TensorDataset(self._dmd._usecols)
        n_train = int(len(data) * train_size / (train_size + val_size))
        n_val = len(data) - n_train
        if n_val > 0:
            return pt.utils.data.TensorDataset(
                data[:n_train][0]
            ), pt.utils.data.TensorDataset(data[n_train:][0])
        else:
            return data, None

    def forward(self, time_indices: pt.Tensor) -> pt.Tensor:
        evals = pt.cat((self._eigvals, self._eigvals[self._conj_indices].conj()))
        evecs = pt.cat(
            (self._eigvecs, self._eigvecs[:, self._conj_indices].conj()), dim=1
        )
        vander = pt.vstack([evals ** n.item() for n in time_indices[0]]).T
        return evecs @ vander

    def train(
        self,
        epochs: int = 1000,
        lr: float = 1.0e-3,
        batch_size: int = None,
        train_size: Union[int, float] = 0.75,
        val_size: Union[int, float] = 0.25,
        normalize_eigvals: bool = False,
        loss_function: Callable = l2_loss,
        scheduler_options: dict = {},
        stopping_options: dict = {},
        loss_key: str = "val_loss",
    ):
        if normalize_eigvals:
            ev = self._eigvals.detach()
            ev /= ev.abs()
            self._eigvals = pt.nn.Parameter(ev)
        optimizer = pt.optim.AdamW(self.parameters(), lr=lr)
        options = {
            key: scheduler_options[key] if key in scheduler_options else val
            for key, val in DEFAULT_SCHEDULER_OPT.items()
        }
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **options
        )
        train_set, val_set = self._create_train_val_split(train_size, val_size)
        batch_size, shuffle = (
            (batch_size, True) if batch_size else (len(train_set), False)
        )
        train_loader = pt.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
        )
        stopper = EarlyStopping(model=self, **stopping_options)
        e, stop = 0, False
        while e < epochs and not stop:
            train_loss = 0.0
            for batch in train_loader:
                pred = self.forward(batch)
                loss = loss_function(
                    self._dmd._dm[:, batch[0]], pred, self._eigvecs, self._eigvals
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self._log["train_loss"].append(train_loss / len(train_loader))
            if val_set is not None:
                with pt.no_grad():
                    batch = val_set[:]
                    self._log["val_loss"].append(
                        loss_function(
                            self._dmd._dm[:, batch[0]],
                            self.forward(batch),
                            self._eigvecs,
                            self._eigvals,
                        ).item()
                    )
            self._log["lr"].append(optimizer.param_groups[0]["lr"])
            val_loss = self._log["val_loss"][-1] if val_set is not None else 0.0
            scheduler.step(self._log[loss_key][-1])
            print(
                "\rEpoch {:4d} - train loss: {:1.6e}, val loss: {:1.6e}, lr: {:1.6e}".format(
                    e, self._log["train_loss"][-1], val_loss, self._log["lr"][-1]
                ),
                end="",
            )
            e += 1
            stop = stopper(self._log[loss_key][-1])

    def partial_reconstruction(self, mode_indices: Set[int]) -> pt.Tensor:
        modes = self.modes
        mode_mask = pt.zeros(modes.shape[1], dtype=modes.dtype)
        mode_indices = pt.tensor(list(mode_indices), dtype=pt.int64)
        mode_mask[mode_indices] = 1.0
        rec = (modes * mode_mask) @ self.dynamics
        if not self._dmd._complex:
            rec = rec.real
        return rec.type(self._dmd._dm.dtype)

    def top_modes(
        self,
        n: int = 10,
        integral: bool = False,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        importance = self.integral_contribution if integral else self.amplitude
        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]), dtype=pt.int64)[
            modes_in_range
        ]
        n = min(n, mode_indices.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]

    @property
    def dmd_init(self) -> DMD:
        return self._dmd

    @property
    def log(self) -> dict:
        return self._log

    @property
    def modes(self) -> pt.Tensor:
        return self.eigvecs / self.amplitude

    @property
    def eigvals(self) -> pt.Tensor:
        ev = self._eigvals.detach()
        return pt.cat((ev, ev[self._conj_indices].conj()))

    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self.eigvals) / self._dmd._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        ev = self._eigvecs.detach()
        return pt.cat((ev, ev[:, self._conj_indices].conj()), dim=1)

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self.eigvals).imag / (2.0 * pi * self._dmd._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self.eigvals) / self._dmd._dt).real

    @property
    def amplitude(self) -> pt.Tensor:
        return self.eigvecs.norm(dim=0)

    @property
    def dynamics(self) -> pt.Tensor:
        vander = pt.vstack([self.eigvals ** n.item() for n in self._dmd._usecols]).T
        return pt.diag(self.amplitude.type(vander.dtype)) @ vander

    @property
    def integral_contribution(self) -> pt.Tensor:
        return self.amplitude**2 * self.dynamics.abs().sum(dim=1)

    @property
    def reconstruction(self) -> pt.Tensor:
        return (self.modes @ self.dynamics).real

    @property
    def reconstruction_error(self) -> pt.Tensor:
        return self._dmd._dm[:, self._dmd._usecols] - self.reconstruction
