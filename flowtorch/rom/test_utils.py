# standard library packages
from time import sleep
# flowtorch packages
from .utils import log_time


def test_log_time():
    @log_time
    def wait_seconds(time_to_wait: float):
        sleep(time_to_wait)
        return {"test": 0}
    log = wait_seconds(0.1)
    assert "execution_time" in log.keys()
    assert "test" in log.keys()
    assert abs(log["execution_time"] - 0.1) < 0.01
