import json
import os
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from spiking_wildfire.config import ROOT


def map_key() -> str:
    env = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if env:
        return env
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("FIRMS_MAP_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("FIRMS_MAP_KEY is missing")


def _get(url: str, key: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "spiking-wildfire"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last).replace(key, "***")) from None


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***")


def availability(sensor: str) -> pd.DataFrame:
    key = map_key()
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/{sensor}"
    raw = _get(url, key)
    text = raw.decode("utf-8", errors="replace")
    if "latitude" not in text.splitlines()[0].lower() and "," not in text.splitlines()[0]:
        raise RuntimeError(_redact(text[:300], key))
    frame = pd.read_csv(pd.io.common.StringIO(text))
    frame.columns = [column.strip().lower() for column in frame.columns]
    return frame


def study_coverage(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if "date" in frame.columns and "sp" in frame.columns:
        dates = pd.to_datetime(frame["date"])
        window = frame[(dates >= start) & (dates <= end)].copy()
        window["date"] = pd.to_datetime(window["date"])
        return window
    if "min_date" in frame.columns:
        return frame
    raise RuntimeError(f"unrecognized availability columns: {list(frame.columns)}")


def missing_study_days(frame: pd.DataFrame, start: str, end: str) -> list[str]:
    if "date" not in frame.columns or "sp" not in frame.columns:
        return []
    covered = set(
        pd.to_datetime(frame.loc[frame["sp"].astype(str).str.lower().isin(["true", "1"]), "date"])
        .dt.strftime("%Y-%m-%d")
    )
    days = pd.date_range(start, end, freq="D")
    return [day.strftime("%Y-%m-%d") for day in days if day.strftime("%Y-%m-%d") not in covered]


def download_firms(cfg: dict, dest: Path) -> None:
    key = map_key()
    dest.mkdir(parents=True, exist_ok=True)
    box = cfg["bbox"]
    area = f"{box['west']},{box['south']},{box['east']},{box['north']}"
    sensor = cfg["sensor"]
    start = date.fromisoformat(cfg["study_start"])
    end = date.fromisoformat(cfg["study_end"])
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=4), end)
        day_range = (chunk_end - cursor).days + 1
        target = dest / f"{sensor}_{cursor.isoformat()}_{day_range}.csv"
        if target.exists() and target.stat().st_size > 0:
            cursor = chunk_end + timedelta(days=1)
            continue
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{key}/{sensor}/{area}/{day_range}/{cursor.isoformat()}"
        )
        raw = _get(url, key)
        text = raw.decode("utf-8", errors="replace")
        first = text.splitlines()[0].lower() if text.splitlines() else ""
        if "latitude" not in first:
            raise RuntimeError(_redact(text[:300], key))
        target.write_text(text, encoding="utf-8")
        print(f"saved {target.name} ({len(text.splitlines()) - 1} rows)", flush=True)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.2)


def parse_acquisition(acq_date, acq_time) -> pd.Series:
    dates = pd.to_datetime(acq_date)
    clock = pd.to_numeric(acq_time, errors="coerce").fillna(0).astype(int)
    hours = clock // 100
    minutes = clock % 100
    return dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def load_events(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"no FIRMS csv files in {raw_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.columns = [column.strip().lower() for column in frame.columns]
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    events["time"] = parse_acquisition(events["acq_date"], events["acq_time"])
    events = events.rename(columns={"latitude": "lat", "longitude": "lon"})
    keep = ["time", "lat", "lon", "frp", "confidence"]
    if "satellite" in events.columns:
        events = events[events["satellite"].astype(str).str.upper().isin(["N", "SNPP", "SUOMI-NPP"])]
    events = events[keep].dropna(subset=["lat", "lon", "time"])
    events["frp"] = pd.to_numeric(events["frp"], errors="coerce")
    events = events.drop_duplicates(subset=["time", "lat", "lon", "frp"])
    return events.sort_values("time").reset_index(drop=True)


def write_availability(sensor: str, start: str, end: str, dest: Path) -> pd.DataFrame:
    frame = availability(sensor)
    dest.parent.mkdir(parents=True, exist_ok=True)
    window = study_coverage(frame, start, end)
    window.to_csv(dest, index=False)
    missing = missing_study_days(frame, start, end)
    if not missing and "min_date" in frame.columns and "date" not in frame.columns:
        earliest = pd.to_datetime(frame["min_date"]).min()
        latest = pd.to_datetime(frame["max_date"]).max()
        if earliest > pd.Timestamp(start) or latest < pd.Timestamp(end):
            missing = [start]
    summary = {
        "sensor": sensor,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_study_days": missing[:20],
        "missing_count": len(missing),
    }
    (dest.parent / "availability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing:
        raise RuntimeError(f"{sensor} is missing {len(missing)} study days, first {missing[0]}")
    return window
