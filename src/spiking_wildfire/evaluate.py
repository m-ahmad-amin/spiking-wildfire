import json
from collections import deque

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from spiking_wildfire.config import resolve
from spiking_wildfire.data.labels import labels_for_split, perimeter_mask
from spiking_wildfire.encode import encode_events, study_hours
from spiking_wildfire.grid import Grid
from spiking_wildfire.lif import LifConfig
from spiking_wildfire.readout import average_precision, f1_at_half, fit_logistic, predict_proba
from spiking_wildfire.stream import (
    detection_matrix,
    forecast_matrix,
    future_mask,
    recent_mask,
    simulate,
    spike_window,
)
from spiking_wildfire.track import RegionMemory, Tracker, region_overlap


def _hour_index(hours: pd.DatetimeIndex, day: str) -> int:
    stamp = pd.Timestamp(day) + pd.Timedelta(hours=23)
    return int(hours.get_loc(stamp))


def _sample(step, hourly, t, horizon, n_lat, n_lon, rng, ratio=5):
    quiet = ~recent_mask(hourly, t, n_lat, n_lon, 24)
    future = future_mask(hourly, t, horizon, n_lat, n_lon)
    pos_i, pos_j = np.where(quiet & future)
    neg_i, neg_j = np.where(quiet & ~future)
    if len(pos_i) == 0:
        return None
    take = min(len(neg_i), ratio * len(pos_i))
    choice = rng.choice(len(neg_i), size=take, replace=False) if take else np.array([], dtype=int)
    ii = np.concatenate([pos_i, neg_i[choice]]) if take else pos_i
    jj = np.concatenate([pos_j, neg_j[choice]]) if take else pos_j
    labels = np.concatenate([np.ones(len(pos_i)), np.zeros(take)])
    return forecast_matrix(step, ii, jj), labels


def _thin(features, labels, cap, rng):
    if len(labels) <= cap:
        return features, labels
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    keep_pos = pos if len(pos) <= cap // 2 else rng.choice(pos, cap // 2, replace=False)
    keep_neg = rng.choice(neg, min(len(neg), cap - len(keep_pos)), replace=False)
    index = np.concatenate([keep_pos, keep_neg])
    return features[index], labels[index]


def _with_cells(events: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    i, j, valid = grid.locate(events["lat"].to_numpy(), events["lon"].to_numpy())
    kept = events.loc[valid].copy()
    kept["i"] = i[valid]
    kept["j"] = j[valid]
    return kept


def evaluate_encoded(
    events: pd.DataFrame,
    spans: pd.DataFrame,
    hourly,
    hours: pd.DatetimeIndex,
    grid: Grid,
    cfg: dict,
    lif_cfg: LifConfig,
    write_frames: bool = False,
) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    n_lat, n_lon = grid.n_lat, grid.n_lon
    winter = cfg["anomaly"]["winter_months"]
    min_months = int(cfg["anomaly"]["min_months"])
    label_tables = {}
    counts = {}
    for name, bounds in cfg["splits"].items():
        table = labels_for_split(events, spans, bounds[0], bounds[1], min_months, winter)
        label_tables[name] = table
        counts[name] = {
            "wildfire_cells": int((table["label"] == 1).sum()) if len(table) else 0,
            "anomaly_cells": int((table["label"] == 0).sum()) if len(table) else 0,
        }
    ends = {name: _hour_index(hours, bounds[1]) for name, bounds in cfg["splits"].items()}
    test_start = int(hours.get_loc(pd.Timestamp(cfg["splits"]["test"][0])))
    stride = int(cfg["forecast_stride_hours"])
    horizons = [int(item) for item in cfg["horizons_hours"]]
    train_x = {horizon: [] for horizon in horizons}
    train_y = {horizon: [] for horizon in horizons}
    det_x = {}
    det_y = {}
    forecast_models = {}
    det_model = None
    det_scaler = None
    test_scores = {horizon: [] for horizon in horizons}
    test_truth = {horizon: [] for horizon in horizons}
    dilate_scores = []
    dilate_truth = []
    tracker = Tracker(cfg["track"]["gate_cells"], cfg["track"]["expire_hours"])
    memory = RegionMemory()
    recent = deque(maxlen=int(cfg["track"]["spike_hours"]))
    iou_values = []
    switches = 0
    updates = 0
    frame_dir = resolve(cfg, "frames")
    frame_times = []
    if write_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
    sparse = cfg["lif"]["mode"] == "sparse"
    device = cfg.get("device")
    for step in simulate(hourly, n_lat, n_lon, lif_cfg, sparse, device=device):
        updates += step.updates
        t = step.t
        in_train = t <= ends["train"]
        if in_train and t % stride == 0:
            for horizon in horizons:
                if t + horizon >= len(hourly):
                    continue
                sample = _sample(step, hourly, t, horizon, n_lat, n_lon, rng)
                if sample is None:
                    continue
                train_x[horizon].append(sample[0])
                train_y[horizon].append(sample[1])
        for name, end_t in ends.items():
            if t != end_t:
                continue
            table = label_tables[name]
            if table.empty:
                continue
            cells = table[["i", "j"]].to_numpy(dtype=int)
            det_x[name] = detection_matrix(step, cells)
            det_y[name] = table["label"].to_numpy(dtype=int)
        if t == ends["train"]:
            det_model, det_scaler = _fit_detection(det_x, det_y, "train", int(cfg["seed"]))
            for horizon in horizons:
                if not train_x[horizon]:
                    continue
                features = np.vstack(train_x[horizon])
                labels = np.concatenate(train_y[horizon])
                features, labels = _thin(features, labels, 300_000, rng)
                if len(np.unique(labels)) < 2:
                    continue
                forecast_models[horizon] = fit_logistic(features, labels, int(cfg["seed"]))
        if t >= test_start and t % stride == 0:
            quiet = ~recent_mask(hourly, t, n_lat, n_lon, 24)
            active = recent_mask(hourly, t, n_lat, n_lon, 24)
            neighbor = binary_dilation(active) & ~active
            for horizon, packed in forecast_models.items():
                if t + horizon >= len(hourly):
                    continue
                ii, jj = np.where(quiet)
                if len(ii) == 0:
                    continue
                scores = predict_proba(packed[0], packed[1], forecast_matrix(step, ii, jj))
                truth = future_mask(hourly, t, horizon, n_lat, n_lon)[ii, jj]
                test_scores[horizon].append(scores)
                test_truth[horizon].append(truth.astype(np.int8))
                if horizon == 24:
                    dilate_scores.append(neighbor[ii, jj].astype(np.float32))
                    dilate_truth.append(truth.astype(np.int8))
        if t >= test_start:
            recent.append(step.spikes.detach().cpu().numpy() > 0)
            id_map = tracker.update(spike_window(recent))
            if (t + 1) % 24 == 0:
                day_mask = perimeter_mask(spans, hours[t], n_lat, n_lon)
                iou, _ignored, _current = region_overlap(id_map, day_mask, {})
                iou_values.append(iou)
                switches += memory.observe(day_mask, id_map)
            if write_frames and det_model is not None and t % int(cfg["frame_stride_hours"]) == 0:
                frame_times.append(
                    _write_frame(
                        frame_dir,
                        grid,
                        hours[t],
                        step,
                        id_map,
                        det_model,
                        det_scaler,
                        forecast_models.get(24),
                    )
                )
    if write_frames:
        (frame_dir / "catalog.json").write_text(
            json.dumps({"frames": frame_times}, indent=2),
            encoding="utf-8",
        )
    from spiking_wildfire.models import save_readouts

    readout_file = None
    if det_model is not None:
        readout_file = str(
            save_readouts(cfg, (det_model, det_scaler), forecast_models.get(24))
        )
    detection = {}
    for name in ("val", "test"):
        if det_model is None or name not in det_x:
            detection[name] = {"average_precision": None, "f1": None}
            continue
        scores = predict_proba(det_model, det_scaler, det_x[name])
        detection[name] = {
            "average_precision": average_precision(det_y[name], scores),
            "f1": f1_at_half(det_y[name], scores),
        }
    forecast = {}
    for horizon in horizons:
        if not test_scores[horizon]:
            forecast[str(horizon)] = None
            continue
        forecast[str(horizon)] = average_precision(
            np.concatenate(test_truth[horizon]),
            np.concatenate(test_scores[horizon]),
        )
    return {
        "counts": counts,
        "detection": detection,
        "forecast_average_precision": forecast,
        "dilation_24h_average_precision": average_precision(
            np.concatenate(dilate_truth), np.concatenate(dilate_scores)
        )
        if dilate_scores
        else None,
        "tracking": {
            "mean_daily_iou": float(np.mean(iou_values)) if iou_values else None,
            "identity_switches": int(switches),
            "days": len(iou_values),
        },
        "cell_updates": int(updates),
        "readouts": readout_file,
        "lif": {
            "tau_hours": lif_cfg.tau_hours,
            "v_th": lif_cfg.v_th,
            "v_reset": lif_cfg.v_reset,
            "refrac_hours": lif_cfg.refrac_hours,
            "couple_weight": lif_cfg.couple_weight,
            "mode": cfg["lif"]["mode"],
            "encoding": cfg["encoding"],
        },
    }


def _fit_detection(det_x, det_y, name, seed):
    if name not in det_x or len(np.unique(det_y[name])) < 2:
        return None, None
    features, labels = _thin(det_x[name], det_y[name], 300_000, np.random.default_rng(seed))
    return fit_logistic(features, labels, seed)


def _write_frame(frame_dir, grid: Grid, when, step, id_map, det_model, det_scaler, forecast_pack):
    v = step.v.detach().cpu().numpy()
    spike = step.spikes.detach().cpu().numpy() > 0
    spatial = step.spatial.detach().cpu().numpy()
    rate24 = step.rate24.detach().cpu().numpy() / 24
    candidates = spike | (v >= 0.25) | (id_map > 0)
    ii, jj = np.where(candidates)
    forecast = np.zeros(len(ii))
    if len(ii) and forecast_pack is not None:
        features = np.column_stack([v[ii, jj], spatial[ii, jj], rate24[ii, jj]])
        forecast = predict_proba(forecast_pack[0], forecast_pack[1], features)
    keep = np.ones(len(ii), dtype=bool)
    if len(ii):
        keep = spike[ii, jj] | (v[ii, jj] >= 0.25) | (id_map[ii, jj] > 0) | (
            (forecast >= 0.55) & (v[ii, jj] >= 0.2)
        )
    ii = ii[keep]
    jj = jj[keep]
    forecast = forecast[keep] if len(forecast) else forecast
    det = np.zeros(len(ii))
    if len(ii) and det_model is not None:
        cells = np.column_stack([ii, jj])
        det = predict_proba(det_model, det_scaler, detection_matrix(step, cells))
    records = []
    for index, (i, j) in enumerate(zip(ii, jj)):
        lat, lon = grid.center(int(i), int(j))
        records.append(
            {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "v": round(float(v[i, j]), 4),
                "spike": int(spike[i, j]),
                "det": round(float(det[index]), 4),
                "track": int(id_map[i, j]),
                "forecast": round(float(forecast[index]), 4) if len(forecast) else 0.0,
            }
        )
    stamp = pd.Timestamp(when).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"time": stamp, "cells": records}
    (frame_dir / f"{stamp.replace(':', '')}.json").write_text(json.dumps(payload), encoding="utf-8")
    return stamp


def load_inputs(cfg: dict):
    from spiking_wildfire.grid import grid_from_config

    grid = grid_from_config(cfg)
    events = pd.read_parquet(resolve(cfg, "events"))
    events["time"] = pd.to_datetime(events["time"])
    spans = pd.read_parquet(resolve(cfg, "spans"))
    spans["start"] = pd.to_datetime(spans["start"])
    spans["end"] = pd.to_datetime(spans["end"])
    events = _with_cells(events, grid)
    hours = study_hours(cfg["study_start"], cfg["study_end"])
    hourly = encode_events(events, grid, hours, cfg["confidence"], cfg["encoding"])
    return events, spans, hourly, hours, grid
