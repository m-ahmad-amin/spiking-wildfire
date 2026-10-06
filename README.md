# spiking-wildfire

A leaky integrate-and-fire grid for NASA FIRMS detections. The scored study is Suomi-NPP over 32.5–42.0°N, 124.5–114.0°W, from 1 Jan 2019 through 31 Dec 2020. Cells are 0.05° (about 5 km).

```
pip install -e ".[dev]"
python scripts/prepare_data.py
python scripts/run_stream.py
python scripts/run_baselines.py
python scripts/run_ablations.py
python scripts/serve_map.py
```

`prepare_data.py` reads `FIRMS_MAP_KEY` from the environment or `.env`. The map is at http://127.0.0.1:8765 after `run_stream.py` has written frames. Fitted detection and forecast readouts are saved to `data/results/readouts.joblib` by `run_stream.py` or `scripts/fit_readouts.py`. `scripts/trim_frames.py` shrinks existing frame files. The map server polls global FIRMS about once an hour.

For the full parameter sweep and ConvLSTM on a GPU, open `notebooks/colab_sweep.ipynb` in Colab, set Runtime → GPU, and run the cells. Results land in `data/results/baselines.json` and `data/results/ablations.json` on Drive.

`python scripts/run_nrt.py` polls the latest global VIIRS day into a sparse LIF. That output is operational. It is not a scored map.

The run writes `data/results/metrics.json` with cell counts. Read these with the numbers:

- August–December 2020 was an extreme season next to a milder training year.
- This box has oil fields, landfills, geothermal sites, and agricultural burning more than gas flares. The anomaly class may be small.
- A 5 km cell can hold a refinery and nearby wildland together.
- Most hours are empty because the satellite passes about twice a day.
- Perimeter polygons are final footprints, painted from start to end. 2019 history polygons cover the fire year, because that layer has no discovery date.
- Global output reuses the study-box weights.
