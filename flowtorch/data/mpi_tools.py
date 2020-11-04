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


def main_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if comm.Get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper


def job_conditional(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "job" in kwargs.keys():
            job = kwargs["job"]
        else:
            print("Warning: function {:s} has no job argument.".format(
                func.__name__))
            job = 0
        if job % comm.Get_size() == comm.Get_rank():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


@main_only
def log_message(message):
    print(message)
