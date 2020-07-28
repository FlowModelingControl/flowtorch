import torch as pt
from flowtorch.data import FOAMDataloader

def test_FOAMDataloader():
    dataloader = FOAMDataloader(".")
    assert dataloader.get_data_matrix().shape == pt.Size([3, 3])
