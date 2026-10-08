import json
from pathlib import Path

from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.data.firms import download_firms, load_events, write_availability
from spiking_wildfire.data.perimeters import download_spans
from spiking_wildfire.grid import grid_from_config


def main():
    cfg = load_config()
    raw = resolve(cfg, "raw_firms")
    print("checking Suomi-NPP availability")
    write_availability(cfg["sensor"], cfg["study_start"], cfg["study_end"], raw.parent / "availability.csv")
    print("downloading Suomi-NPP")
    download_firms(cfg, raw)
    events = load_events(raw)
    events_path = resolve(cfg, "events")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    print(f"events {len(events)}")
    print("downloading national perimeters")
    spans = download_spans(cfg, grid_from_config(cfg), resolve(cfg, "spans"))
    print(
        "2019 perimeter cells use the fire year because that history layer has no discovery date. "
        f"span rows {len(spans)}"
    )
    summary = {
        "events": int(len(events)),
        "span_rows": int(len(spans)),
        "years": {str(k): int(v) for k, v in spans["year"].value_counts().items()} if len(spans) else {},
    }
    Path(resolve(cfg, "results")).mkdir(parents=True, exist_ok=True)
    (resolve(cfg, "results") / "prepare.json").write_text(json.dumps(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
