r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation solder. Currently, only the ESI branch
of OpenFOAM is supported (v1912, v2006).
"""

from .dataloader import Dataloader
import torch as pt

class FOAMDataloader(Dataloader):
    r"""

    """
    def __init__(self, path):
        r"""[summary]

        :param path: [description]
        :type path: [type]
        """

    def get_data_matrix(self, bounds: list = [], field_names: list = []) -> pt.Tensor:
        r"""

        """
        pass