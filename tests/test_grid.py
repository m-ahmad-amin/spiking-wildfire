import pandas as pd

from spiking_wildfire.encode import confidence_weight, encode_events, study_hours
from spiking_wildfire.grid import Grid


def test_grid_corners():
    grid = Grid(32.5, 42.0, -124.5, -114.0, 0.05)
    assert grid.n_lat == 190
    assert grid.n_lon == 210
    i, j, valid = grid.locate([32.5, 32.55, 42.0, 42.1], [-124.5, -124.5, -114.0, -114.0])
    assert (int(i[0]), int(j[0]), bool(valid[0])) == (0, 0, True)
    assert int(i[1]) == 1
    assert (int(i[2]), int(j[2]), bool(valid[2])) == (189, 209, True)
    assert not bool(valid[3])


def test_log_frp_encoding_lands_in_the_right_hour():
    grid = Grid(32.5, 42.0, -124.5, -114.0, 0.05)
    hours = study_hours("2019-01-01", "2019-01-01")
    events = pd.DataFrame(
        {
            "time": [pd.Timestamp("2019-01-01 13:45"), pd.Timestamp("2019-01-01 13:10")],
            "lat": [32.52, 32.52],
            "lon": [-124.47, -124.47],
            "frp": [10.0, 0.0],
            "confidence": ["h", "l"],
        }
    )
    mapping = {"l": 0.3, "n": 0.7, "h": 1.0}
    assert confidence_weight("n", mapping) == 0.7
    assert confidence_weight("missing", mapping) is None
    bins = encode_events(events, grid, hours, mapping, "log_frp")
    hour = 13
    assert len(bins[hour][0]) == 1
    assert bins[hour][2][0] > 0
    assert bins[12][0].size == 0
