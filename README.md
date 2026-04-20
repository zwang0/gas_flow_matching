# Conditional Flow Matching for Gas Trajectories

This project contains a conditional flow-matching model for generating full concentration trajectories `c_{time}` at each 3D location.

## Project structure

- `main.py`: main CLI entrypoint with `train` and `generate` commands.
- `fm_gas/data_utils.py`: dataset loading and alignment utilities.
- `fm_gas/features.py`: conditioning and normalization utilities.
- `fm_gas/model.py`: model definition and ODE sampling.
- `fm_gas/train.py`: training loop.
- `fm_gas/generate.py`: generation/export logic.
- `scripts/train.py`: dedicated training script.
- `scripts/sample.py`: dedicated sampling script.
- `flow_matching_gas.py`: backward-compatible wrapper to `main.py`.

## Data used

- Trajectory data (positions + concentration trajectories):
  - `data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv`
- Sensor time-series conditioning:
  - `data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv`
- Sensor coordinates:
  - `data/sensor_coords.csv`
- Inlet/outlet coordinates:
  - `data/inlet_outlet_coords.csv`

The default setup is inlet `18`, outlet `09`, flow `50 sccm`.

## Install

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python -m pip install -r requirements.txt
```

## Train

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py train \
  --traj-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv \
  --sensor-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --inlet-id 18 \
  --outlet-id 9 \
  --flow-sccm 50 \
  --epochs 300 \
  --batch-size 256 \
  --lr 2e-4 \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt
```

Alternative:

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python scripts/train.py \
  --inlet-id 18 --outlet-id 9 --flow-sccm 50
```

## Generate trajectories

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py generate \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt \
  --traj-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv \
  --sensor-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --inlet-id 18 \
  --outlet-id 9 \
  --flow-sccm 50 \
  --num-steps 200 \
  --output-csv outputs/generated_trajectories_09_18_09_50sccm.csv
```

Alternative:

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python scripts/sample.py \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt \
  --inlet-id 18 --outlet-id 9 --flow-sccm 50
```

The output CSV format is:
- First three columns: `x_m`, `y_m`, `z_m`
- Remaining columns: generated `c_{time}` trajectory values

## Model summary

- Training objective: conditional flow matching
- State variable: full concentration trajectory vector at one 3D point
- Conditions:
  - sensor time-series values (all sensors, all times)
  - sensor coordinates
  - inlet coordinate
  - outlet coordinate
  - flow speed
  - target 3D point position (`x_m, y_m, z_m`)

## Fixed-XYZ process formulation

Each training sample corresponds to one fixed spatial point `(x_m, y_m, z_m)`.
Only concentration evolves over time, so the flow-matching state is the trajectory vector:

- `x1 = [c_t0, c_t1, ..., c_tT]`

The point coordinates are used as conditioning inputs, not generated outputs.

## Sensor constraints

Constraint behavior is controlled by `--constraint-mode`:

- `none`: no explicit sensor constraint
- `soft`: add train-time sensor consistency loss
- `hard`: apply sampling-time projection to sensor targets
- `hybrid`: both soft (train) and hard (sample)

Useful flags:

- `--lambda-sensor`: weight of sensor loss in training
- `--lambda-sensor-warmup-epochs`: warmup for sensor-loss weighting
- `--projection-alpha`: projection strength (0 to 1)
- `--projection-every`: apply projection every N Euler steps
- `--sensor-map-k`: map each sensor to k nearest trajectory points

### Train with hybrid constraints

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py train \
  --traj-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv \
  --sensor-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --inlet-id 18 \
  --outlet-id 9 \
  --flow-sccm 50 \
  --epochs 300 \
  --batch-size 256 \
  --lr 2e-4 \
  --constraint-mode hybrid \
  --lambda-sensor 1.0 \
  --lambda-sensor-warmup-epochs 30 \
  --sensor-map-k 1 \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt
```

### Generate with hybrid constraints (full field)

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py generate \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt \
  --traj-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv \
  --sensor-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv \
  --sensor-coords-csv data/sensor_coords.csv \
  --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
  --inlet-id 18 \
  --outlet-id 9 \
  --flow-sccm 50 \
  --constraint-mode hybrid \
  --projection-alpha 1.0 \
  --projection-every 1 \
  --sensor-map-k 1 \
  --num-steps 200 \
  --output-csv outputs/generated_trajectories_09_18_09_50sccm.csv
```

### Generate one fixed-point process

Provide a `positions_csv` containing one row with `x_m,y_m,z_m`.
The output will contain one row and all `c_t` columns for that fixed point.

```bash
cd /home/wjq8vw/projects/gas_flow_matching
/home/wjq8vw/miniforge3/bin/python main.py generate \
  --checkpoint checkpoints/flow_matcher_09_18_09_50sccm.pt \
  --traj-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv \
  --sensor-csv data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv \
  --positions-csv data/one_point.csv \
  --constraint-mode soft \
  --num-steps 200 \
  --output-csv outputs/generated_one_point_09_18_09_50sccm.csv
```
