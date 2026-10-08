import json

from spiking_wildfire.config import load_config, resolve
from spiking_wildfire.evaluate import evaluate_encoded, load_inputs
from spiking_wildfire.lif import LifConfig


def main():
    cfg = load_config()
    cfg = dict(cfg)
    cfg["frame_stride_hours"] = 10**9
    events, spans, hourly, hours, grid = load_inputs(cfg)
    result = evaluate_encoded(
        events,
        spans,
        hourly,
        hours,
        grid,
        cfg,
        LifConfig.from_dict(cfg["lif"]),
        write_frames=False,
    )
    dest = resolve(cfg, "results")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"readouts": result.get("readouts"), "detection": result["detection"]}, indent=2))


if __name__ == "__main__":
    main()
