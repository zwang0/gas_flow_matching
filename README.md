# Autoregressive Flow Matching for Gas Sensors

This project contains an autoregressive conditional flow-matching model for generating full 3D concentration trajectories over time.

## Project structure

- `main.py`: main CLI entrypoint with `train` and `generate` commands.
- `fm_gas/data_utils.py`: trajectory + conditioning loading utilities.
- `fm_gas/model.py`: spatially-aware vector field model and Euler sampler.
- `fm_gas/train.py`: training loop.
- `fm_gas/generate.py`: full-field generation/export logic.

## Data used

- Binned trajectories (target):
  - `data/*_binned.csv` (columns `x_m,y_m,z_m,c_{time}`)
- Surface averages (conditioning):
  - `data/*_surface_averages.csv`
- Sensor positions:
  - `data/sensor_coords.csv` (columns `sensor,x,y,z` for 12 sensors)
- Inlet/outlet coordinates:
  - `data/inlet_outlet_coords.csv`

## Install

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python -m pip install -r requirements.txt
```

## Train

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py train \
  --data-dir data \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --history-k 1 \
  --epochs 300 \
  --batch-size 256 \
  --lr 2e-4 \
  --checkpoint checkpoints/flow_matcher.pt
```

## Generate trajectories

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py generate \
  --checkpoint checkpoints/flow_matcher.pt \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --binned-csv data/Gas_3D_sim08_19_09_50sccm_binned.csv \
  --surface-avg-csv data/Gas_3D_sim08_19_09_50sccm_surface_averages.csv \
  --num-steps 50 \
  --output-csv outputs/generated_surface_trajectory.csv
```

The output CSV format is:
- First three columns: `x_m`, `y_m`, `z_m`
- Remaining columns: `c_{time}` trajectory values

## Model summary

- Training objective: autoregressive conditional flow matching
- State variable: full field concentration vector per timestep
- Conditioning:
  - surface averages history window of size `k`
  - full surface averages time series
  - inlet/outlet coordinates and sccm parsed from each trajectory filename
  - sensor positions as part of the global condition
