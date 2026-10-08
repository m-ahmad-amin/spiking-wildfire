import json
import os
from copy import deepcopy

import pandas as pd
import torch

from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.encode import encode_events, study_hours
from spiking_wildfire.evaluate import _with_cells, evaluate_encoded
from spiking_wildfire.grid import grid_from_config
from spiking_wildfire.lif import LifConfig, torch_device


def _variant(cfg, lif_updates, encoding=None):
    nxt = deepcopy(cfg)
    nxt["lif"].update(lif_updates)
    if encoding is not None:
        nxt["encoding"] = encoding
    nxt["forecast_stride_hours"] = max(int(cfg["forecast_stride_hours"]), 24)
    nxt["frame_stride_hours"] = 10**9
    return nxt


def main():
    cfg = load_config()
    device = torch_device(os.environ.get("SPIKING_DEVICE"))
    cfg["device"] = str(device)
    print(f"device {device}", flush=True)
    grid = grid_from_config(cfg)
    events = pd.read_parquet(resolve(cfg, "events"))
    events["time"] = pd.to_datetime(events["time"])
    spans = pd.read_parquet(resolve(cfg, "spans"))
    spans["start"] = pd.to_datetime(spans["start"])
    spans["end"] = pd.to_datetime(spans["end"])
    events = _with_cells(events, grid)
    hours = study_hours(cfg["study_start"], cfg["study_end"])
    variants = [
        ("default", {}, None),
        ("tau_12", {"tau_hours": 12}, None),
        ("tau_48", {"tau_hours": 48}, None),
        ("threshold_0.5", {"v_th": 0.5}, None),
        ("threshold_2", {"v_th": 2}, None),
        ("reset_0.2", {"v_reset": 0.2}, None),
        ("refrac_0", {"refrac_hours": 0}, None),
        ("refrac_6", {"refrac_hours": 6}, None),
        ("no_coupling", {"couple_weight": 0}, None),
        ("binary", {}, "binary"),
        ("count", {}, "count"),
        ("sparse_cpu", {"mode": "sparse"}, None),
    ]
    rows = []
    for name, updates, encoding in variants:
        trial = _variant(cfg, updates, encoding)
        if name == "sparse_cpu":
            trial["device"] = "cpu"
            trial["lif"]["mode"] = "sparse"
        else:
            trial["device"] = cfg["device"]
            if device.type == "cuda":
                trial["lif"]["mode"] = "dense"
        hourly = encode_events(events, grid, hours, trial["confidence"], trial["encoding"])
        result = evaluate_encoded(
            events,
            spans,
            hourly,
            hours,
            grid,
            trial,
            LifConfig.from_dict(trial["lif"]),
            write_frames=False,
        )
        rows.append(
            {
                "name": name,
                "detection_test_ap": result["detection"]["test"]["average_precision"],
                "forecast_24h_ap": result["forecast_average_precision"].get("24"),
                "cell_updates": result["cell_updates"],
                "device": trial["device"],
                "mode": trial["lif"]["mode"],
            }
        )
        print(rows[-1], flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dest = resolve(cfg, "results") / "ablations.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
