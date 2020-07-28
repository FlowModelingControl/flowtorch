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

    def _read_file_header(self, file_path):
        pass

    def _read_volScalarField(self, file_path):
        pass

    def _read_volVectorField(self, file_path):
        pass

    def get_data_matrix(self, bounds: list = [], field_names: list = []) -> pt.Tensor:
        r"""

        """
        return pt.ones(3, 3)


class FOAMCase():
    """Class to access and parse OpenFOAM cases.
    """
    def __init__(self, path):
        pass

    def write_times():
        """Extract the write times from the folder names.
        """
        pass

    def write_format():
        """Determine the encoding of mesh and field data.

        The encoding may be ascii or binary.
        """
        pass