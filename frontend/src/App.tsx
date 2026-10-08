import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchCatalog,
  fetchFrame,
  refreshWorld,
  stepHour,
  warmup,
  type Frame,
} from "./api";
import { FireMap } from "./components/FireMap";
import { ProgressRail, type ProgressKind } from "./components/ProgressRail";

type Layers = {
  spikes: boolean;
  membrane: boolean;
  detect: boolean;
  tracks: boolean;
  forecast: boolean;
};

const LAYER_LABELS: Record<keyof Layers, string> = {
  spikes: "Activity",
  membrane: "Intensity",
  detect: "Detection",
  tracks: "Tracks",
  forecast: "Forecast",
};

const DEFAULT_LAYERS: Layers = {
  spikes: true,
  membrane: true,
  detect: true,
  tracks: false,
  forecast: true,
};

export default function App() {
  const [frames, setFrames] = useState<string[]>([]);
  const [index, setIndex] = useState(0);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [layers, setLayers] = useState<Layers>(DEFAULT_LAYERS);
  const [mode, setMode] = useState("California 2020");
  const [status, setStatus] = useState("Loading map data…");
  const [busy, setBusy] = useState(false);
  const [progressKind, setProgressKind] = useState<ProgressKind | null>("startup");
  const [center, setCenter] = useState<[number, number]>([37.5, -119.5]);
  const [zoom, setZoom] = useState(6);
  const playRef = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;

    (async () => {
      setProgressKind("startup");
      setStatus("Loading map data…");

      try {
        const catalog = await fetchCatalog();
        if (!alive) return;
        const list = catalog.frames || [];
        setFrames(list);

        if (list.length) {
          const first = await fetchFrame(list[0]);
          if (!alive) return;
          setFrame(first);
          setStatus(formatPlaybackStatus(first, DEFAULT_LAYERS));
        } else {
          setStatus("Playback data is not available yet.");
        }
      } catch (error) {
        if (!alive) return;
        setProgressKind("reconnect");
        setStatus(
          error instanceof Error
            ? error.message
            : "Could not reach the server. Retrying…",
        );
        window.setTimeout(() => {
          if (alive) window.location.reload();
        }, 5000);
        return;
      } finally {
        if (alive) setProgressKind(null);
      }

      // Warm live services in the background; do not block the map.
      void warmup().catch(() => {
        /* playback already works; live actions retry on click */
      });
    })();

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (playRef.current) window.clearInterval(playRef.current);
    };
  }, []);

  const shown = useMemo(
    () => (frame ? countVisible(frame, layers) : 0),
    [frame, layers],
  );

  async function loadIndex(next: number) {
    if (!frames[next]) return;
    setMode("California 2020");
    setCenter([37.5, -119.5]);
    setZoom(6);
    try {
      const payload = await fetchFrame(frames[next]);
      setIndex(next);
      setFrame(payload);
      setStatus(formatPlaybackStatus(payload, layers));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not load that hour.");
    }
  }

  function stopPlay() {
    if (playRef.current) {
      window.clearInterval(playRef.current);
      playRef.current = null;
    }
  }

  function onPlay() {
    stopPlay();
    setMode("California 2020");
    setCenter([37.5, -119.5]);
    setZoom(6);
    let cursor = index;
    playRef.current = window.setInterval(() => {
      cursor += 1;
      if (cursor >= frames.length) {
        stopPlay();
        return;
      }
      void loadIndex(cursor);
    }, 700);
  }

  async function onStep(reset = false) {
    stopPlay();
    setBusy(true);
    setProgressKind("step");
    setMode("Live hour-by-hour");
    setStatus(reset ? "Restarting the live window…" : "Advancing one hour…");
    setCenter([37.5, -119.5]);
    setZoom(6);
    try {
      const payload = await stepHour(reset);
      setFrame(payload);
      if (payload.done) setStatus("End of this window. Use Reset to start again.");
      else setStatus(`${payload.time}  ·  hour ${payload.hour} of ${payload.hours}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not run that step.");
    } finally {
      setBusy(false);
      setProgressKind(null);
    }
  }

  async function onWorld() {
    stopPlay();
    setBusy(true);
    setProgressKind("world");
    setMode("Global live view");
    setStatus("Fetching the latest satellite detections…");
    setCenter([20, 0]);
    setZoom(2);
    try {
      const payload = await refreshWorld();
      setFrame(payload);
      if (payload.error) setStatus(payload.error);
      else {
        setStatus(
          `${payload.time || "Latest"}  ·  ${payload.detections || 0} detections`,
        );
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not refresh the world view.");
    } finally {
      setBusy(false);
      setProgressKind(null);
    }
  }

  return (
    <div className="shell">
      <FireMap frame={frame} layers={layers} center={center} zoom={zoom} />
      <aside className="panel">
        <p className="brand">Wildfire LIF</p>
        <h1>{mode}</h1>
        <p className="status">{status}</p>
        <p className="meta">{shown} locations shown</p>
        <input
          className="slider"
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          value={index}
          onChange={(event) => {
            stopPlay();
            void loadIndex(Number(event.target.value));
          }}
        />
        <div className="layers">
          {(Object.keys(layers) as (keyof Layers)[]).map((key) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={(event) =>
                  setLayers((current) => ({ ...current, [key]: event.target.checked }))
                }
              />
              {LAYER_LABELS[key]}
            </label>
          ))}
        </div>
        <div className="actions">
          <button type="button" onClick={onPlay} disabled={busy || !frames.length}>
            Play
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => void loadIndex(index)}
            disabled={busy || !frames.length}
          >
            California
          </button>
          <button type="button" className="ghost" onClick={() => void onStep(false)} disabled={busy}>
            Step one hour
          </button>
          <button type="button" className="ghost" onClick={() => void onStep(true)} disabled={busy}>
            Reset step
          </button>
          <button type="button" className="ghost" onClick={() => void onWorld()} disabled={busy}>
            World refresh
          </button>
        </div>
        <p className="note">
          Play through the 2020 California season, step through a live mid-August window, or refresh
          the latest global satellite heat detections.
        </p>
        <ProgressRail
          kind={progressKind || "startup"}
          active={progressKind !== null}
          title={
            progressKind === "world"
              ? "Updating live detections"
              : progressKind === "step"
                ? "Running live simulation"
                : progressKind === "reconnect"
                  ? "Reconnecting"
                  : "Getting things ready"
          }
        />
      </aside>
    </div>
  );
}

function formatPlaybackStatus(frame: Frame, layers: Layers) {
  return `${frame.time || "Frame"}  ·  ${countVisible(frame, layers)} locations`;
}

function countVisible(frame: Frame, layers: Layers) {
  return (frame.cells || []).filter((cell) => {
    const interesting = cell.spike || cell.v >= 0.25 || cell.track > 0;
    const forecastHit = layers.forecast && cell.forecast >= 0.55 && cell.v >= 0.2;
    return interesting || forecastHit;
  }).length;
}
