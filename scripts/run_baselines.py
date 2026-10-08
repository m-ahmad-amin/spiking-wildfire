import json

import pandas as pd

from spiking_wildfire.baselines import convlstm_forecast, lstm_forecast, tabular_forecast
from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.evaluate import load_inputs


def main():
    cfg = load_config()
    events, _spans, hourly, hours, grid = load_inputs(cfg)
    train_start = cfg["splits"]["train"][0]
    train_end = cfg["splits"]["train"][1] + " 23:00"
    test_start = cfg["splits"]["test"][0]
    test_end = cfg["splits"]["test"][1] + " 23:00"
    train_end_index = int(hours.get_loc(pd.Timestamp(train_end)))
    result = {
        "gradient_boosting_test_average_precision": tabular_forecast(
            events, train_start, train_end, test_start, test_end, 24, int(cfg["seed"])
        ),
        "lstm_test_average_precision": lstm_forecast(
            events, int(cfg["seed"]), train_start, train_end, test_start, test_end
        ),
        "convlstm_test_average_precision": convlstm_forecast(
            hourly, grid.n_lat, grid.n_lon, int(cfg["seed"]), train_end_index
        ),
    }
    dest = resolve(cfg, "results") / "baselines.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
