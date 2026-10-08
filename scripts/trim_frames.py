import json
from pathlib import Path

from spiking_wildfire.config import load_config, resolve


def trim_frame(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kept = []
    for cell in payload.get("cells", []):
        if (
            cell.get("spike")
            or cell.get("v", 0) >= 0.25
            or cell.get("track", 0)
            or (cell.get("forecast", 0) >= 0.55 and cell.get("v", 0) >= 0.2)
        ):
            kept.append(cell)
    payload["cells"] = kept
    path.write_text(json.dumps(payload), encoding="utf-8")
    return len(kept)


def main():
    cfg = load_config()
    frame_dir = resolve(cfg, "frames")
    catalog = json.loads((frame_dir / "catalog.json").read_text(encoding="utf-8"))
    total = 0
    for stamp in catalog["frames"]:
        path = frame_dir / f"{stamp.replace(':', '')}.json"
        if path.exists():
            total += trim_frame(path)
    print(json.dumps({"frames": len(catalog["frames"]), "cells_kept": total}))


if __name__ == "__main__":
    main()
