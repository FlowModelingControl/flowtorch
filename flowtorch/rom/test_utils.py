# standard library packages
from time import sleep
from pytest import raises
# flowtorch packages
from .utils import log_time, check_positive_integer


def test_log_time():
    @log_time
    def wait_seconds(time_to_wait: float):
        sleep(time_to_wait)
        return {"test": 0}
    log = wait_seconds(0.1)
    assert "execution_time" in log.keys()
    assert "test" in log.keys()
    assert abs(log["execution_time"] - 0.1) < 0.01


def test_check_positive_integer():
    with raises(ValueError):
        check_positive_integer(1.0, "name")
    with raises(ValueError):
        check_positive_integer(0, "name")
    check_positive_integer(1, "name")
