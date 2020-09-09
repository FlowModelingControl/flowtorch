from functools import wraps
from mpi4py import MPI


comm = MPI.COMM_WORLD


def main_bcast(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if comm.Get_rank() == 0:
            result = func(*args, **kwargs)
        else:
            result = None
        return comm.bcast(result, root=0)
    return wrapper
