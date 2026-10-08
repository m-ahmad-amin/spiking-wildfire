import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

from spiking_wildfire.grid import Grid

HISTORY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/"
    "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query"
)
WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)


def _query(url: str, params: dict) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(full, headers={"User-Agent": "spiking-wildfire"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return payload
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last)) from None


def _pages(url: str, where: str, box: dict, out_fields: str) -> list[dict]:
    features = []
    offset = 0
    page = 40
    geometry = f"{box['west']},{box['south']},{box['east']},{box['north']}"
    while True:
        payload = _query(
            url,
            {
                "where": where,
                "geometry": geometry,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": out_fields,
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
                "resultOffset": str(offset),
                "resultRecordCount": str(page),
            },
        )
        batch = payload.get("features", [])
        features.extend(batch)
        print(f"perimeters {offset + len(batch)}", flush=True)
        exceeded = bool(payload.get("exceededTransferLimit"))
        if not batch or (len(batch) < page and not exceeded):
            break
        offset += len(batch)
    return features


def _esri_time(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)) and value > 10_000_000_000:
        return pd.to_datetime(value, unit="ms", utc=True).tz_localize(None)
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_localize(None)


def _cells_for_polygon(poly, grid: Grid) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = poly.bounds
    j0 = max(0, int(np.floor((minx - grid.west) / grid.step)))
    j1 = min(grid.n_lon, int(np.floor((maxx - grid.west) / grid.step)) + 1)
    i0 = max(0, int(np.floor((miny - grid.south) / grid.step)))
    i1 = min(grid.n_lat, int(np.floor((maxy - grid.south) / grid.step)) + 1)
    if i0 >= i1 or j0 >= j1:
        return []
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    south = grid.south + ii * grid.step
    west = grid.west + jj * grid.step
    boxes = shapely.box(west, south, west + grid.step, south + grid.step)
    hit = shapely.intersects(poly, boxes)
    return list(zip(ii[hit].ravel().tolist(), jj[hit].ravel().tolist()))


def _append_span(rows, poly, grid: Grid, start, end, name: str, year: int):
    if start is None or end is None or end < start:
        return
    for i, j in _cells_for_polygon(poly, grid):
        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "start": pd.Timestamp(start).normalize(),
                "end": pd.Timestamp(end).normalize(),
                "name": name,
                "year": int(year),
            }
        )


def spans_from_history(features: list[dict], grid: Grid) -> list[dict]:
    rows = []
    for feature in features:
        props = feature.get("properties") or {}
        year = props.get("FIRE_YEAR_INT")
        if year is None:
            continue
        geometry = feature.get("geometry")
        if not geometry:
            continue
        start = pd.Timestamp(year=int(year), month=1, day=1)
        end = pd.Timestamp(year=int(year), month=12, day=31)
        _append_span(rows, shape(geometry), grid, start, end, props.get("INCIDENT") or "", int(year))
    return rows


def spans_from_wfigs(features: list[dict], grid: Grid) -> list[dict]:
    rows = []
    for feature in features:
        props = feature.get("properties") or {}
        geometry = feature.get("geometry")
        if not geometry:
            continue
        start = _esri_time(props.get("attr_FireDiscoveryDateTime"))
        end = _esri_time(props.get("attr_FireOutDateTime"))
        if end is None:
            end = _esri_time(props.get("attr_ContainmentDateTime"))
        if end is None:
            end = _esri_time(props.get("attr_ControlDateTime"))
        if start is None:
            continue
        if end is None:
            end = pd.Timestamp(year=int(start.year), month=12, day=31)
        name = props.get("poly_IncidentName") or ""
        _append_span(rows, shape(geometry), grid, start, end, name, int(start.year))
    return rows


def download_spans(cfg: dict, grid: Grid, dest: Path) -> pd.DataFrame:
    box = cfg["bbox"]
    history = _pages(HISTORY_URL, "FIRE_YEAR_INT=2019", box, "FIRE_YEAR_INT,INCIDENT")
    wfigs = _pages(
        WFIGS_URL,
        "attr_FireDiscoveryDateTime >= DATE '2020-01-01' AND attr_FireDiscoveryDateTime < DATE '2021-01-01'",
        box,
        "attr_FireDiscoveryDateTime,attr_FireOutDateTime,attr_ContainmentDateTime,attr_ControlDateTime,poly_IncidentName",
    )
    rows = spans_from_history(history, grid) + spans_from_wfigs(wfigs, grid)
    frame = pd.DataFrame(rows, columns=["i", "j", "start", "end", "name", "year"])
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["i", "j", "start", "end", "name"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(f"history features {len(history)}, wfigs features {len(wfigs)}, span rows {len(frame)}")
    return frame
