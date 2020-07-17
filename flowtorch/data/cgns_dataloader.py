r"""Implementation of a concrete :class:`Dataloader` class.

The :class:`CGNSDataloader` class allows to load fields from
CGNS files located in a folder.
"""

from .dataloader import Dataloader
import torch as pt

class CGNSDataloader(Dataloader):
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