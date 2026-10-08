# Results summary

Study box: 32.5–42.0°N, 124.5–114.0°W. Sensor: VIIRS Suomi-NPP. Grid: 0.05°. Period: 2019-01-01 to 2020-12-31.

## Scores (`metrics.json`)

- Train cells: 1129 wildfire, 359 standing-heat
- Val cells: 600 wildfire, 0 standing-heat (no detection score)
- Test cells: 1596 wildfire, 54 standing-heat
- Detection test AP / F1: 0.992 / 0.982
- Forecast AP (new activity): 12h 0.032, 24h 0.058, 72h 0.101
- Dilation 24h AP: 0.016
- Tracking mean daily IoU: 0.045; identity switches: 205
- Sparse cell updates: 39,223,021

## Baselines (`baselines.json`)

- Gradient boosting 24h AP: 0.459
- LSTM 24h AP: 0.975 (busy-cell subset; not the same quiet-cell task)
- ConvLSTM 24h AP: 0.032

## Ablations (`ablations.json`)

- Best 24h forecast: reset_0.2 (AP 0.069)
- No coupling drops 24h forecast to 0.032
- Sparse CPU: ~39M updates; dense CUDA: ~700M updates

## Artifacts

- LIF parameters: `configs/default.yaml`
- Fitted readouts: `data/results/readouts.joblib` (after `python scripts/fit_readouts.py` or `run_stream.py`)
- Map: `python scripts/serve_map.py` → http://127.0.0.1:8765
- World poller runs hourly inside the map server (operational, not scored)

## Write-up notes

- August–December 2020 was extreme relative to training.
- Standing-heat class is small in this box.
- Cells are ~5 km; final perimeters are painted across their date span.
- Global map reuses California readouts and is not scored.
