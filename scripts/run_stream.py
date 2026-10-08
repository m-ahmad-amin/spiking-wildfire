import json

from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.evaluate import evaluate_encoded, load_inputs
from spiking_wildfire.lif import LifConfig

NOTES = [
    "August to December 2020 was an extreme season. Training mostly sees 2019 and early 2020.",
    "Anomaly cells are standing heat in this box, not a large gas-flare class. Counts are above.",
    "Cells are about 5 km, so a refinery and nearby wildland can share one cell.",
    "Most hours are empty because Suomi-NPP passes about twice a day. The map mostly shows decay.",
    "Final perimeters are painted across their date span, so early detection and forecasts look late.",
    "2019 history polygons are labeled for the whole fire year. They have no daily outline.",
    "Global mode, when run, reuses these weights and is not a scored map.",
]


def main():
    cfg = load_config()
    events, spans, hourly, hours, grid = load_inputs(cfg)
    result = evaluate_encoded(
        events,
        spans,
        hourly,
        hours,
        grid,
        cfg,
        LifConfig.from_dict(cfg["lif"]),
        write_frames=True,
    )
    result["notes"] = NOTES
    dest = resolve(cfg, "results")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
