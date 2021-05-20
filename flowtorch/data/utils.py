"""Collection of utilities realted to data and dataloaders."""


from typing import Tuple


def format_byte_size(size: int) -> Tuple[float, str]:
    """Convert number of bytes into human-readable format.

    The function is based on `this <https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb>` Stackoverflow question.

    :param size: size in bytes
    :type path: int
    :return: converted size corresponding unit
    :rtype: tuple(float, str)

    """
    exponent_labels = {0: "b", 1: "Kb", 2: "Mb", 3: "Gb", 4: "Tb", 5: "Pt"}
    exponent = 0
    while size > 1024:
        size /= 1024
        exponent += 1
    return size, exponent_labels[exponent]
