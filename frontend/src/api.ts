export type FrameCell = {
  lat: number;
  lon: number;
  v: number;
  spike: number;
  det: number;
  track: number;
  forecast: number;
};

export type Frame = {
  time?: string | null;
  cells: FrameCell[];
  done?: boolean;
  hour?: number;
  hours?: number;
  detections?: number;
  ingested_at?: string | null;
  operational?: boolean;
  world?: boolean;
  error?: string | null;
  note?: string;
  readouts_loaded?: boolean;
};

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function friendlyError(error: unknown): Error {
  if (error instanceof DOMException && error.name === "AbortError") {
    return new Error("This is taking longer than expected. Please try again.");
  }
  if (error instanceof TypeError) {
    return new Error(
      "Could not reach the server. It may still be starting — try again in a moment.",
    );
  }
  if (error instanceof Error) return error;
  return new Error("Something went wrong. Please try again.");
}

async function getJson<T>(url: string, options?: { timeoutMs?: number; retries?: number }): Promise<T> {
  const timeoutMs = options?.timeoutMs ?? 120_000;
  const retries = options?.retries ?? 2;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) {
        throw new Error("The server could not complete that request.");
      }
      return (await response.json()) as T;
    } catch (error) {
      lastError = friendlyError(error);
      if (attempt < retries) await sleep(800 * (attempt + 1));
    } finally {
      window.clearTimeout(timer);
    }
  }

  throw lastError ?? new Error("Something went wrong. Please try again.");
}

export function warmup() {
  return getJson<{ ok: boolean; message: string; frames: boolean; readouts: boolean }>(
    "/api/warmup",
    { timeoutMs: 30_000, retries: 3 },
  );
}

export function fetchStatus() {
  return getJson<{
    frames: boolean;
    warm: boolean;
    live_ready: boolean;
    readouts: boolean;
    poller: boolean;
    world_ingested_at: string | null;
  }>("/api/status", { timeoutMs: 15_000, retries: 2 });
}

export function fetchCatalog() {
  return getJson<{ frames: string[]; error?: string }>("/api/catalog", {
    timeoutMs: 30_000,
    retries: 3,
  });
}

export function fetchFrame(stamp: string) {
  return getJson<Frame>(`/frames/${stamp.replaceAll(":", "")}.json`, {
    timeoutMs: 30_000,
    retries: 2,
  });
}

export function stepHour(reset = false) {
  return getJson<Frame>(`/api/step${reset ? "?reset=1" : ""}`, {
    timeoutMs: 180_000,
    retries: 1,
  });
}

export function refreshWorld() {
  return getJson<Frame>("/api/world?refresh=1", { timeoutMs: 180_000, retries: 1 });
}
