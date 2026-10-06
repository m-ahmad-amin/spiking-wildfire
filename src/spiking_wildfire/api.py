import json
import os
import threading
import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from spiking_wildfire.config import ROOT, load_config
from spiking_wildfire.live import LiveWindow
from spiking_wildfire.models import load_readouts, readout_path
from spiking_wildfire.nrt import WorldMonitor

FRAME_DIR = ROOT / "data" / "frames"
DIST_DIR = ROOT / "frontend" / "dist"
STATE = {
    "window": None,
    "world": None,
    "poller": None,
    "warm": False,
    "warming_live": False,
}


def _warm_live_window() -> None:
    try:
        if STATE["window"] is None:
            STATE["window"] = LiveWindow(load_config())
        STATE["warm"] = True
    except Exception as exc:
        print(f"live warm failed: {exc}", flush=True)
    finally:
        STATE["warming_live"] = False


def create_app() -> FastAPI:
    app = FastAPI(title="Wildfire LIF")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/warmup")
    def warmup():
        cfg = load_config()
        load_readouts(cfg)
        frames_ok = (FRAME_DIR / "catalog.json").exists()
        # Warm the live window in the background so the map can open immediately.
        if STATE["window"] is None and not STATE["warming_live"]:
            STATE["warming_live"] = True
            threading.Thread(target=_warm_live_window, daemon=True).start()
        elif STATE["window"] is not None:
            STATE["warm"] = True
        return {
            "ok": True,
            "warm": STATE["warm"],
            "frames": frames_ok,
            "readouts": readout_path(cfg).exists(),
            "live_ready": STATE["window"] is not None,
            "message": "ready",
        }

    @app.get("/api/status")
    def status():
        cfg = load_config()
        world = STATE["world"]
        return {
            "frames": (FRAME_DIR / "catalog.json").exists(),
            "live_ready": STATE["window"] is not None,
            "warm": STATE["warm"],
            "readouts": readout_path(cfg).exists(),
            "world_ingested_at": None
            if world is None or world.last_ingest is None
            else world.last_ingest.isoformat() + "Z",
            "poller": STATE["poller"] is not None and STATE["poller"].is_alive(),
        }

    @app.get("/api/catalog")
    def catalog():
        path = FRAME_DIR / "catalog.json"
        if not path.exists():
            return JSONResponse(
                {"frames": [], "error": "no frames yet; run scripts/run_stream.py"},
                status_code=200,
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/step")
    def step(reset: int = Query(0)):
        window = STATE["window"]
        if window is None or reset == 1:
            window = LiveWindow()
            STATE["window"] = window
        return window.step()

    @app.get("/api/world")
    def world(refresh: int = Query(0)):
        monitor = STATE["world"]
        if monitor is None:
            monitor = WorldMonitor(load_config())
            STATE["world"] = monitor
        force = refresh == 1 or monitor.last_ingest is None
        return monitor.refresh(force=force)

    @app.get("/frames/{name}")
    def frame(name: str):
        path = FRAME_DIR / name
        if not path.exists():
            return JSONResponse({"error": "missing frame"}, status_code=404)
        return FileResponse(path, media_type="application/json")

    if DIST_DIR.exists():
        assets = DIST_DIR / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str = ""):
            candidate = DIST_DIR / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(DIST_DIR / "index.html")

    return app


def _poll_loop(minutes: int):
    # Let the UI and catalog settle before the first FIRMS pull.
    time.sleep(90)
    while True:
        try:
            world = STATE["world"]
            if world is None:
                world = WorldMonitor(load_config())
                STATE["world"] = world
            summary = world.refresh(force=True)
            print(
                f"world poll detections={summary.get('detections')} "
                f"active={summary.get('active_cells')} error={summary.get('error')}",
                flush=True,
            )
        except Exception as exc:
            print(f"world poll failed: {exc}", flush=True)
        time.sleep(max(minutes, 10) * 60)


def main():
    import uvicorn

    cfg = load_config()
    minutes = int(cfg.get("world_poll_minutes", 60))
    poller = threading.Thread(target=_poll_loop, args=(minutes,), daemon=True)
    STATE["poller"] = poller
    poller.start()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    print(f"api at http://{host}:{port}  world poll every {minutes} min", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
