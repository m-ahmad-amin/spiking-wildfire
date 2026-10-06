import numpy as np
import pandas as pd

from spiking_wildfire.grid import Grid


def confidence_weight(value, mapping: dict) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    if key not in mapping:
        return None
    return float(mapping[key])


def encode_events(
    events: pd.DataFrame,
    grid: Grid,
    hours: pd.DatetimeIndex,
    mapping: dict,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if events.empty:
        empty = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
        return [empty for _ in range(len(hours))]

    frame = events.copy()
    weights = frame["confidence"].map(lambda value: confidence_weight(value, mapping))
    kept = weights.notna() & frame["frp"].notna() & (frame["frp"] >= 0)
    frame = frame.loc[kept].copy()
    if frame.empty:
        empty = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
        return [empty for _ in range(len(hours))]
    frame["weight"] = weights.loc[kept].astype(float)
    i, j, valid = grid.locate(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    frame = frame.loc[valid].copy()
    frame["i"] = i[valid]
    frame["j"] = j[valid]
    frame["hour"] = frame["time"].dt.floor("h")
    start = hours[0]
    frame = frame[(frame["hour"] >= start) & (frame["hour"] <= hours[-1])]
    frame["t"] = ((frame["hour"] - start) / pd.Timedelta(hours=1)).astype(int)

    if mode == "binary":
        frame["value"] = 1.0
        grouped = frame.groupby(["t", "i", "j"], as_index=False)["value"].max()
    elif mode == "count":
        frame["value"] = 1.0
        grouped = frame.groupby(["t", "i", "j"], as_index=False)["value"].sum()
    elif mode == "log_frp":
        frame["value"] = np.log1p(frame["frp"].to_numpy()) * frame["weight"].to_numpy()
        grouped = frame.groupby(["t", "i", "j"], as_index=False)["value"].sum()
    else:
        raise ValueError(f"unknown encoding {mode}")

    empty = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
    by_hour = {int(hour): part for hour, part in grouped.groupby("t", sort=False)}
    bins: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(len(hours)):
        part = by_hour.get(t)
        if part is None:
            bins.append(empty)
            continue
        bins.append(
            (
                part["i"].to_numpy(dtype=int),
                part["j"].to_numpy(dtype=int),
                part["value"].to_numpy(dtype=float),
            )
        )
    return bins


def study_hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
