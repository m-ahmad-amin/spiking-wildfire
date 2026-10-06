from pathlib import Path

import joblib

from spiking_wildfire.config import resolve


def readout_path(cfg: dict) -> Path:
    return resolve(cfg, "results") / "readouts.joblib"


def save_readouts(cfg: dict, detection, forecast_24) -> Path:
    path = readout_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detection": detection,
        "forecast_24": forecast_24,
        "lif": cfg["lif"],
        "encoding": cfg["encoding"],
        "confidence": cfg["confidence"],
    }
    joblib.dump(payload, path)
    return path


def load_readouts(cfg: dict):
    path = readout_path(cfg)
    if not path.exists():
        return None
    return joblib.load(path)
