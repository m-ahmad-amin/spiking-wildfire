from collections import deque
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from spiking_wildfire.data.firms import _get, map_key, parse_acquisition
from spiking_wildfire.lif import LifConfig
from spiking_wildfire.models import load_readouts
from spiking_wildfire.readout import predict_proba


def _tiles(step_lat=45, step_lon=60):
    for south in range(-90, 90, step_lat):
        for west in range(-180, 180, step_lon):
            yield west, south, min(west + step_lon, 180), min(south + step_lat, 90)


def poll_day(sensor: str, day: date) -> pd.DataFrame:
    key = map_key()
    frames = []
    for west, south, east, north in _tiles():
        area = f"{west},{south},{east},{north}"
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{key}/{sensor}/{area}/1/{day.isoformat()}"
        )
        raw = _get(url, key)
        text = raw.decode("utf-8", errors="replace")
        first = text.splitlines()[0].lower() if text.splitlines() else ""
        if "latitude" not in first:
            raise RuntimeError(text[:240].replace(key, "***"))
        frame = pd.read_csv(pd.io.common.StringIO(text))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "frp", "confidence", "time"])
    events = pd.concat(frames, ignore_index=True)
    events.columns = [column.strip().lower() for column in events.columns]
    events["time"] = parse_acquisition(events["acq_date"], events["acq_time"])
    keep = ["time", "lat", "lon", "frp"]
    renamed = events.rename(columns={"latitude": "lat", "longitude": "lon"})
    if "confidence" in renamed.columns:
        keep.append("confidence")
    return renamed[keep]


def step_sparse(
    events: pd.DataFrame,
    cfg: LifConfig,
    state: dict[tuple[int, int], float] | None = None,
    hours_elapsed: float = 1.0,
    step: float = 0.05,
    confidence: dict | None = None,
):
    state = dict(state or {})
    fired: set[tuple[int, int]] = set()
    decay = math_decay(cfg, hours_elapsed)
    for cell in list(state):
        state[cell] *= decay
    if events.empty:
        return prune(state, fired), fired
    weights = confidence or {}
    for row in events.itertuples(index=False):
        lat = float(row.lat)
        lon = float(row.lon)
        frp = float(getattr(row, "frp", 0) or 0)
        weight = 1.0
        if weights and hasattr(row, "confidence"):
            key = str(getattr(row, "confidence")).strip().lower()
            mapped = weights.get(key)
            if mapped is None:
                continue
            weight = float(mapped)
        i = int(np.floor((lat + 90) / step))
        j = int(np.floor((lon + 180) / step))
        value = state.get((i, j), 0.0) + float(np.log1p(max(frp, 0))) * weight
        if value >= cfg.v_th:
            fired.add((i, j))
            state[(i, j)] = cfg.v_reset
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    neighbor = (i + di, j + dj)
                    state[neighbor] = state.get(neighbor, 0.0) + cfg.couple_weight / 8
        else:
            state[(i, j)] = value
    return prune(state, fired), fired


def math_decay(cfg: LifConfig, hours_elapsed: float) -> float:
    if hours_elapsed <= 0:
        return 1.0
    return float(np.exp(-hours_elapsed / cfg.tau_hours))


def prune(state: dict[tuple[int, int], float], fired: set[tuple[int, int]]):
    return {cell: value for cell, value in state.items() if value > 1e-3 or cell in fired}


class WorldMonitor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lif = LifConfig.from_dict(cfg["lif"])
        self.state: dict[tuple[int, int], float] = {}
        self.fired: set[tuple[int, int]] = set()
        self.spike_hist: deque[set[tuple[int, int]]] = deque(maxlen=168)
        self.last_ingest: datetime | None = None
        self.last_day: date | None = None
        self.last_detections = 0
        self.error: str | None = None
        self.readouts = load_readouts(cfg)

    def refresh(self, force: bool = False) -> dict:
        day = date.today() - timedelta(days=1)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if (
            not force
            and self.last_ingest is not None
            and (now - self.last_ingest) < timedelta(minutes=50)
            and self.last_day == day
        ):
            return self.snapshot()
        try:
            frames = [poll_day(sensor, day) for sensor in self.cfg["nrt_sensors"]]
            events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not events.empty:
                events = events.drop_duplicates(subset=["time", "lat", "lon", "frp"])
            hours = 1.0
            if self.last_ingest is not None:
                hours = max((now - self.last_ingest).total_seconds() / 3600.0, 1.0)
            self.state, self.fired = step_sparse(
                events,
                self.lif,
                state=self.state if self.last_day == day else {},
                hours_elapsed=hours,
                confidence=self.cfg.get("confidence"),
            )
            ticks = max(1, int(round(hours)))
            for _ in range(ticks):
                self.spike_hist.append(set(self.fired))
            self.last_detections = int(len(events))
            self.last_ingest = now
            self.last_day = day
            self.error = None
        except Exception as exc:
            self.error = str(exc)
        return self.snapshot()

    def _rates(self, cell: tuple[int, int]) -> tuple[float, float, float]:
        recent24 = list(self.spike_hist)[-24:]
        rate24 = sum(1 for bucket in recent24 if cell in bucket) / max(len(recent24), 1)
        rate7 = sum(1 for bucket in self.spike_hist if cell in bucket) / max(len(self.spike_hist), 1)
        neighbor = 0.0
        i, j = cell
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor += sum(1 for bucket in recent24 if (i + di, j + dj) in bucket)
        neighbor = neighbor / (8 * max(len(recent24), 1))
        return rate24, rate7, neighbor

    def snapshot(self) -> dict:
        cells = []
        active = list(self.state.keys())
        if not active:
            return self._payload(cells)
        det_model = None
        det_scaler = None
        forecast_pack = None
        if self.readouts:
            det_model, det_scaler = self.readouts.get("detection") or (None, None)
            forecast_pack = self.readouts.get("forecast_24")
        features_det = []
        features_fc = []
        for cell in active:
            value = self.state[cell]
            rate24, rate7, neighbor = self._rates(cell)
            active_days = min(30.0, rate7 * 30)
            spatial = 0.0
            i, j = cell
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    spatial += self.state.get((i + di, j + dj), 0.0)
            spatial = spatial * self.lif.couple_weight / 8
            features_det.append([value, rate24, rate7, neighbor, active_days])
            features_fc.append([value, spatial, rate24])
        det_scores = np.zeros(len(active))
        fc_scores = np.zeros(len(active))
        if det_model is not None:
            det_scores = predict_proba(det_model, det_scaler, np.asarray(features_det))
        if forecast_pack is not None:
            fc_scores = predict_proba(forecast_pack[0], forecast_pack[1], np.asarray(features_fc))
        for index, cell in enumerate(active):
            i, j = cell
            value = self.state[cell]
            cells.append(
                {
                    "lat": round((i + 0.5) * 0.05 - 90, 4),
                    "lon": round((j + 0.5) * 0.05 - 180, 4),
                    "v": round(float(value), 4),
                    "spike": int(cell in self.fired),
                    "det": round(float(det_scores[index]), 4),
                    "track": 0,
                    "forecast": round(float(fc_scores[index]), 4),
                }
            )
        return self._payload(cells)

    def _payload(self, cells: list[dict]) -> dict:
        return {
            "time": self.last_day.isoformat() if self.last_day else None,
            "ingested_at": self.last_ingest.isoformat() + "Z" if self.last_ingest else None,
            "cells": cells,
            "detections": self.last_detections,
            "active_cells": len(self.state),
            "world": True,
            "operational": True,
            "readouts_loaded": self.readouts is not None,
            "note": "Operational global view. Not scored. California readouts applied when available.",
            "error": self.error,
        }
