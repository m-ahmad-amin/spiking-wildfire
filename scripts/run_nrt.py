import json
from datetime import date, timedelta

import pandas as pd

from spiking_wildfire.config import load_config
from spiking_wildfire.lif import LifConfig
from spiking_wildfire.nrt import WorldMonitor, poll_day, step_sparse


def main():
    cfg = load_config()
    monitor = WorldMonitor(cfg)
    payload = monitor.refresh(force=True)
    print(json.dumps({k: payload[k] for k in ("time", "ingested_at", "detections", "active_cells", "error", "note")}))


if __name__ == "__main__":
    main()
