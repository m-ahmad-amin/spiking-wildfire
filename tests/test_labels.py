import pandas as pd

from spiking_wildfire.data.labels import labels_for_split


def test_train_labels_do_not_see_the_test_period():
    times = [
        "2019-01-15",
        "2019-02-15",
        "2019-03-15",
        "2019-04-15",
        "2020-08-15",
        "2020-09-15",
        "2020-10-15",
        "2020-11-15",
        "2020-12-15",
    ]
    events = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "i": [0] * 9,
            "j": [0] * 9,
        }
    )
    spans = pd.DataFrame(columns=["i", "j", "start", "end"])
    train = labels_for_split(events, spans, "2019-01-01", "2020-05-31", 4, [12, 1, 2])
    test = labels_for_split(events, spans, "2020-08-01", "2020-12-31", 4, [12, 1, 2])
    assert list(train["label"]) == [0]
    assert list(test["label"]) == [0]
    burned = pd.DataFrame(
        {
            "i": [0],
            "j": [0],
            "start": [pd.Timestamp("2019-01-01")],
            "end": [pd.Timestamp("2019-12-31")],
        }
    )
    wildfire = labels_for_split(events, burned, "2019-01-01", "2020-05-31", 4, [12, 1, 2])
    assert list(wildfire["label"]) == [1]


def test_validation_window_cannot_invent_an_anomaly():
    events = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-06-01", "2020-07-01"]),
            "i": [1, 1],
            "j": [1, 1],
        }
    )
    spans = pd.DataFrame(columns=["i", "j", "start", "end"])
    val = labels_for_split(events, spans, "2020-06-01", "2020-07-31", 4, [12, 1, 2])
    assert val.empty
