import numpy as np
import pandas as pd

from spiking_wildfire.baselines import _forecast_rows, _quiet_future


def test_quiet_cell_with_a_later_detection_is_positive():
    times = np.array(
        [np.datetime64("2020-08-01T00", "h"), np.datetime64("2020-08-03T00", "h")]
    )
    hour = np.datetime64("2020-08-02T00", "h")
    assert _quiet_future(times, hour, 24) is True


def test_cell_active_now_is_not_sampled():
    times = np.array([np.datetime64("2020-08-02T00", "h")])
    hour = np.datetime64("2020-08-02T00", "h")
    assert _quiet_future(times, hour, 24) is None


def test_forecast_rows_include_both_classes():
    events = pd.DataFrame(
        {
            "i": [1, 1, 4],
            "j": [2, 2, 5],
            "time": pd.to_datetime(["2020-08-01 00:00", "2020-08-03 00:00", "2020-08-01 00:00"]),
            "frp": [4.0, 4.0, 1.0],
        }
    )
    packed = _forecast_rows(events, "2020-08-02", "2020-08-02", 24, 0, 100)
    assert packed is not None
    labels = set(packed[1].tolist())
    assert labels == {0, 1}
