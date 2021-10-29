"""POD implementation"""

import torch as pt


class POD():
    """Implementation of proper orthogonal decomposition (PCA).
    """
    def __init__(self, data: pt.Tensor):
        """Constructor method of the :class:`PCA` class.

        :param data: snapshot data matrix
        :type data: pt.Tensor
        """        
        self._data = data

    def compute_decomposition(self, mode="snapshot", solver="svd", device="cpu", precision="single"):
        if mode == "snapshot":
            data_matrix = self._data.get_data_matrix()
            if not data_matrix.device == device:
                data_matrix.to(device=device)
            cov = pt.mm(data_matrix.T, data_matrix) / (data_matrix.shape[0] - 1)
            if solver == "svd":
                U, S, VT = pt.svd(cov)
                self._U = U
                self._S = S
                self._VT = VT