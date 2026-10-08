import { useEffect, useState } from "react";

export type ProgressKind = "startup" | "step" | "world" | "reconnect";

const SCRIPTS: Record<ProgressKind, string[]> = {
  startup: [
    "Opening the map",
    "Loading California playback",
    "Preparing the timeline",
    "Checking live services",
    "Ready when you are",
  ],
  reconnect: [
    "Reaching the server",
    "Checking map data",
    "Preparing California playback",
    "Loading the first hour",
    "Finishing setup",
  ],
  step: [
    "Connecting to the server",
    "Loading the August fire window",
    "Advancing the simulation one hour",
    "Scoring active cells",
    "Updating the map",
  ],
  world: [
    "Connecting to the server",
    "Requesting satellite detections",
    "Filtering recent heat signals",
    "Updating the live map",
    "Preparing the global view",
  ],
};

type Props = {
  kind: ProgressKind;
  active: boolean;
  title?: string;
};

export function ProgressRail({ kind, active, title }: Props) {
  const steps = SCRIPTS[kind];
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setIndex(0);
      return;
    }
    setIndex(0);
    const timer = window.setInterval(() => {
      setIndex((value) => Math.min(value + 1, steps.length - 1));
    }, 2200);
    return () => window.clearInterval(timer);
  }, [active, kind, steps.length]);

  if (!active) return null;

  return (
    <div className="progress-rail" role="status" aria-live="polite">
      <p className="progress-title">{title || "Working"}</p>
      <ol className="progress-steps">
        {steps.map((label, stepIndex) => {
          const state =
            stepIndex < index ? "done" : stepIndex === index ? "current" : "todo";
          return (
            <li key={label} className={`progress-step ${state}`}>
              <span className="progress-mark" aria-hidden="true" />
              <span>{label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
