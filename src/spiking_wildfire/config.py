from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]
