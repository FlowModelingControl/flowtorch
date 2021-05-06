# third party packages
import pytest
import torch as pt
# flowtorch packages
from flowtorch.data import mask_box


def test_mask_box_1d():
    vertices = pt.linspace(0.0, 5.0, 6)
    mask = mask_box(vertices, [2.0], [4.0])
    true_mask = pt.tensor([False, False, True, True, True, False])
    assert pt.all(mask == true_mask)


def test_mask_box_2d():
    vertices = pt.tensor(
        [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 2], [3, 4]]
    )
    mask = mask_box(vertices, [1, 2], [2, 3])
    true_mask = pt.tensor([True, True, False, True, False, False, False])
    assert pt.all(mask == true_mask)


def test_mask_box_3d():
    vertices = pt.tensor(
        [[1, 2, 1], [1, 3, 2], [1, 4, 2], [2, 3, 2], [2, 4, 1], [3, 2, 1], [3, 4, 2]]
    )
    mask = mask_box(vertices, [1, 2, 0], [2, 3, 2])
    true_mask = pt.tensor([True, True, False, True, False, False, False])
    assert pt.all(mask == true_mask)
