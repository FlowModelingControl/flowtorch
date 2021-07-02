# third party packages
import pytest
# flowtorch packages
from flowtorch.data.utils import format_byte_size


def test_byte_formatting():
    size, unit = format_byte_size(1e3)
    assert size == 1e3 and unit == "b"
    size, unit = format_byte_size(1e4)
    assert size == 1e4/1024 and unit == "Kb"
    size, unit = format_byte_size(1e7)
    assert size == 1e7/1024**2 and unit == "Mb"
