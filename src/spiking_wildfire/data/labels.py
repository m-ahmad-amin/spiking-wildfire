import numpy as np
import pandas as pd


def wildfire_cells(spans: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[tuple[int, int]]:
    if spans.empty:
        return set()
    overlap = spans[(spans["end"] >= start) & (spans["start"] <= end)]
    return set(zip(overlap["i"].astype(int), overlap["j"].astype(int)))


def anomaly_cells(
    months_by_cell: dict[tuple[int, int], set[int]],
    wildfire: set[tuple[int, int]],
    min_months: int,
    winter: set[int],
) -> set[tuple[int, int]]:
    found = set()
    for cell, months in months_by_cell.items():
        if cell in wildfire:
            continue
        if len(months) >= min_months and months & winter:
            found.add(cell)
    return found


def months_in_window(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    if events.empty:
        return {}
    window = events[(events["time"] >= start) & (events["time"] <= end)]
    grouped: dict[tuple[int, int], set[int]] = {}
    for i, j, month in zip(window["i"], window["j"], window["time"].dt.month):
        grouped.setdefault((int(i), int(j)), set()).add(int(month))
    return grouped


def labels_for_split(
    events: pd.DataFrame,
    spans: pd.DataFrame,
    start: str,
    end: str,
    min_months: int,
    winter: list[int],
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)
    wildfire = wildfire_cells(spans, start_ts, end_ts.normalize())
    anomalies = anomaly_cells(
        months_in_window(events, start_ts, end_ts),
        wildfire,
        min_months,
        set(winter),
    )
    rows = [{"i": i, "j": j, "label": 1} for i, j in sorted(wildfire)]
    rows.extend({"i": i, "j": j, "label": 0} for i, j in sorted(anomalies))
    return pd.DataFrame(rows, columns=["i", "j", "label"])


def perimeter_mask(spans: pd.DataFrame, day, n_lat: int, n_lon: int) -> np.ndarray:
    mask = np.zeros((n_lat, n_lon), dtype=bool)
    if spans.empty:
        return mask
    stamp = pd.Timestamp(day).normalize()
    part = spans[(spans["start"] <= stamp) & (spans["end"] >= stamp)]
    if part.empty:
        return mask
    mask[part["i"].to_numpy(dtype=int), part["j"].to_numpy(dtype=int)] = True
    return mask
