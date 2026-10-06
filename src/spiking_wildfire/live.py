from collections import deque

import numpy as np
import pandas as pd

from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.encode import encode_events
from spiking_wildfire.grid import grid_from_config
from spiking_wildfire.lif import LifConfig
from spiking_wildfire.models import load_readouts
from spiking_wildfire.readout import predict_proba
from spiking_wildfire.stream import detection_matrix, forecast_matrix, simulate, spike_window
from spiking_wildfire.track import Tracker

WINDOW_START = "2020-08-16 00:00"
WINDOW_END = "2020-08-18 23:00"


class LiveWindow:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.grid = grid_from_config(self.cfg)
        self.lif = LifConfig.from_dict(self.cfg["lif"])
        self.readouts = load_readouts(self.cfg)
        events = pd.read_parquet(resolve(self.cfg, "events"))
        events["time"] = pd.to_datetime(events["time"])
        start = pd.Timestamp(WINDOW_START)
        end = pd.Timestamp(WINDOW_END)
        self.events = events[(events["time"] >= start) & (events["time"] <= end)].copy()
        self.hours = pd.date_range(start, end, freq="h")
        self.hourly = encode_events(
            self.events,
            self.grid,
            self.hours,
            self.cfg["confidence"],
            self.cfg["encoding"],
        )
        self.reset()

    def reset(self):
        sparse = self.cfg["lif"]["mode"] == "sparse"
        self._steps = simulate(
            self.hourly,
            self.grid.n_lat,
            self.grid.n_lon,
            self.lif,
            sparse,
        )
        self.index = -1
        self.tracker = Tracker(self.cfg["track"]["gate_cells"], self.cfg["track"]["expire_hours"])
        self.recent = deque(maxlen=int(self.cfg["track"]["spike_hours"]))
        self.finished = False

    def step(self) -> dict:
        if self.finished:
            return {"done": True, "time": WINDOW_END, "cells": []}
        try:
            state = next(self._steps)
        except StopIteration:
            self.finished = True
            return {"done": True, "time": WINDOW_END, "cells": []}
        self.index += 1
        spikes = state.spikes.detach().cpu().numpy() > 0
        self.recent.append(spikes)
        tracks = self.tracker.update(spike_window(self.recent))
        return self._frame(state, tracks)

    def _frame(self, state, tracks) -> dict:
        membrane = state.v.detach().cpu().numpy()
        spikes = state.spikes.detach().cpu().numpy() > 0
        keep = spikes | (membrane >= 0.25) | (tracks > 0)
        ii, jj = np.where(keep)
        det = np.zeros(len(ii))
        forecast = np.zeros(len(ii))
        if len(ii) and self.readouts:
            det_pack = self.readouts.get("detection")
            fc_pack = self.readouts.get("forecast_24")
            cells = np.column_stack([ii, jj])
            if det_pack and det_pack[0] is not None:
                det = predict_proba(det_pack[0], det_pack[1], detection_matrix(state, cells))
            if fc_pack is not None:
                forecast = predict_proba(fc_pack[0], fc_pack[1], forecast_matrix(state, ii, jj))
        cells = []
        for index, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
            lat, lon = self.grid.center(int(i), int(j))
            cells.append(
                {
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "v": round(float(membrane[i, j]), 4),
                    "spike": int(spikes[i, j]),
                    "det": round(float(det[index]), 4),
                    "track": int(tracks[i, j]),
                    "forecast": round(float(forecast[index]), 4),
                }
            )
        when = pd.Timestamp(self.hours[self.index]).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "time": when,
            "cells": cells,
            "done": False,
            "hour": self.index + 1,
            "hours": len(self.hours),
            "readouts_loaded": self.readouts is not None,
        }
