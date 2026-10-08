# Spiking Wildfire

**Spiking Wildfire** is a **neurocomputing** project: a **spiking neural network (SNN)** built from **leaky integrate-and-fire (LIF)** neurons on a geographic grid, driven by NASA FIRMS heat detections. **[▶ Live Demo](https://spiking-wildfire.onrender.com/)**

Each ~5 km cell is a LIF unit. Satellite fire radiative power is encoded as input current. Membrane potential integrates over hours, leaks between overflights, couples weakly to neighbors, and emits a **spike** when threshold is crossed. The grid is the network; time is event-driven and sparse by design.

Readouts on LIF state support wildfire-vs-anomaly detection, track association, and short-horizon forecast. The scored study is Suomi-NPP VIIRS over California (**32.5–42.0°N, 124.5–114.0°W**, **2019–2020**). A global near-real-time view reuses the same SNN machinery (operational, not scored).

![architecture](https://res.cloudinary.com/dzzrxqiho/image/upload/v1791423947/spiking-wildfire-architecture_yemw9r.jpg)

## Why SNN / LIF here

Wildfire satellite data is **sparse in space and time**. Most cells are quiet most hours. That matches what SNNs are good at:

- **Event-driven compute**: update cells that received current (or neighbors under coupling), not the full dense grid every hour
- **Temporal memory without heavy recurrence**: membrane τ carries recent heat across gaps between VIIRS passes
- **Explicit dynamics**: threshold, reset, refractory period, and spatial kernel are parameters you can ablate, not opaque hidden state
- **Streaming = replay**: the same LIF step runs offline over the study archive and online on NRT feeds

This repo treats the fire map as a **spatially coupled LIF sheet**, then attaches lightweight logistic readouts. Baselines (boosting, LSTM, ConvLSTM) sit beside it so the neurocomputing claim is measurable, not decorative.

---

## Features

### Neurocomputing core

- **LIF neuron per grid cell** with τ, v_th, v_reset, refractory hours (`configs/default.yaml`)
- **Spatial coupling kernel** so activity can spread to neighbors without a separate diffusion model
- **Sparse vs dense modes**: sparse cell updates by default (~18× fewer ops than dense on this study); dense path for CUDA / ablation parity
- **Log-FRP encoding** (and sweeps) from FIRMS detections into input current
- **No surrogate gradients on the LIF**: dynamics are forward simulation; classifiers are separate readouts on causal LIF features

### Downstream tasks on spike / membrane state

- **Detection**: wildfire vs standing heat (industry, landfills, agricultural burning, and similar)
- **Tracking**: hourly connected components linked across time
- **Forecast**: new activity at 12 / 24 / 72 h vs fire-mask dilation
- **Baselines & ablations**: boosting, LSTM, ConvLSTM, and LIF/encoding parameter sweeps (incl. Colab GPU notebook)

### Product surface

- React + Leaflet map: California 2020 playback, live LIF hour step, global NRT refresh
- FastAPI backend serving frames, `/api/step`, `/api/world`, and background FIRMS polling

![screenshot](https://res.cloudinary.com/dzzrxqiho/image/upload/v1791426373/Group_3_3_dttmvc.png)

---

## Study scope

| Item | Value |
|------|--------|
| Model | Spatially coupled LIF SNN (sparse mode default) |
| Sensor (study) | VIIRS Suomi-NPP SP |
| Region | California box above |
| Grid | 0.05° (~5 km); one LIF neuron per cell |
| Period | 1 Jan 2019 – 31 Dec 2020 |
| Split | Train -> May 2020 · Val Jun–Jul · Test Aug–Dec 2020 |
| Encoding | Log fire radiative power (FRP) as synaptic / input current |

Config: `configs/default.yaml`. Metrics and write-up notes: [`data/results/SUMMARY.md`](data/results/SUMMARY.md).

---

## Repository layout

```
configs/                LIF and study parameters (neurocomputing knobs)
src/spiking_wildfire/   LIF core, encoding, stream, readouts, FastAPI
frontend/               React map UI → frontend/dist
scripts/                Prep, LIF stream, serve, NRT, baselines, ablations
notebooks/              Colab GPU sweep for SNN vs baselines
data/                   Events, frames, results (large; not all committed)
tests/                  Incl. streaming-equals-replay checks
```

---

## Setup

### Requirements

- Python **3.11+**
- Node.js **18+** (UI build)
- A [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) map key

### Install

```bash
pip install -e ".[dev]"
cd frontend && npm install && npm run build && cd ..
```

```bash
FIRMS_MAP_KEY=your_key_here
```

(Use `.env` or export in the shell.)

### Prepare data and run the LIF stream

```bash
python scripts/prepare_data.py   # FIRMS + perimeters → events / labels
python scripts/run_stream.py     # LIF SNN stream, metrics, frames, readouts
```

Readouts land in `data/results/readouts.joblib`.  
Optional: `python scripts/fit_readouts.py`, `python scripts/trim_frames.py`.

### Launch the app

```bash
python scripts/serve_api.py
```

Open **http://127.0.0.1:8765**.

- Playback: saved 2020 California LIF frames
- **Step**: live LIF on 16–18 Aug 2020
- **World refresh**: global VIIRS NRT into the same sparse LIF

UI hot reload (API already running):

```bash
cd frontend && npm run dev
```

### Other entry points

| Command | Purpose |
|---------|---------|
| `python scripts/run_nrt.py` | Global NRT sparse LIF (operational) |
| `python scripts/run_baselines.py` | Boosting / LSTM / ConvLSTM vs LIF features |
| `python scripts/run_ablations.py` | LIF and encoding neurocomputing sweeps |
| `notebooks/colab_sweep.ipynb` | GPU sweep → `baselines.json` / `ablations.json` |

---

## Reading the results

See `data/results/` (`metrics.json`, `baselines.json`, `ablations.json`). Keep in mind:

- Aug–Dec 2020 was extreme relative to training
- Standing-heat labels in this box are rare
- A 5 km LIF cell can mix industry and wildland
- Most hours are empty: VIIRS passes about twice a day (sparsity is the point)
- Perimeter polygons are final footprints painted across their date span
- Global map **reuses California readouts** and is **not** scored

---

## License / citation

Research prototype in neuromorphic / neurocomputing applied to Earth observation. Cite NASA FIRMS for the underlying detections if you publish from this work.
