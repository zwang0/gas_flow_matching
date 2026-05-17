import os

import numpy as np
import pandas as pd
import torch

from .data_utils import build_global_cond_from_filename, load_trajectory_tensor
from .features import default_sensor_positions_path, load_sensor_positions
from .model import AutoregressiveFlowMatcher, euler_sample


def _load_history_from_surface_csv(path: str, history_k: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Surface averages file {path} must include time and sensor columns.")
    time_values = df.iloc[:, 0].to_numpy(dtype=np.float32)
    series = df.iloc[:history_k, 1:].to_numpy(dtype=np.float32)
    if series.shape[0] < history_k:
        raise ValueError("Not enough timesteps in the init surface averages file.")
    return series, time_values


def generate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.sensor_coords_csv:
        sensor_positions = load_sensor_positions(args.sensor_coords_csv)
    else:
        sensor_positions = load_sensor_positions(default_sensor_positions_path(args.data_dir))

    model = AutoregressiveFlowMatcher(
        sensor_positions=torch.from_numpy(sensor_positions).to(device),
        global_cond_dim=int(ckpt["global_cond_mean"].shape[-1]),
        history_k=int(ckpt["history_k"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        num_heads=int(ckpt["num_heads"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.init_surface_csv:
        history, time_values = _load_history_from_surface_csv(args.init_surface_csv, int(ckpt["history_k"]))
        global_cond = build_global_cond_from_filename(args.init_surface_csv, args.inlet_outlet_coords_csv)
    else:
        batch = load_trajectory_tensor(
            args.data_dir,
            inlet_outlet_coords_csv=args.inlet_outlet_coords_csv,
            max_files=1,
        )
        history = batch.trajectories[0, : int(ckpt["history_k"])]
        time_values = batch.time_values
        global_cond = batch.global_cond[0]

    traj_mean = np.asarray(ckpt["traj_mean"], dtype=np.float32)
    traj_std = np.asarray(ckpt["traj_std"], dtype=np.float32)
    traj_mean_flat = traj_mean.squeeze(0).squeeze(0)
    traj_std_flat = traj_std.squeeze(0).squeeze(0)

    history_norm = (history - traj_mean_flat) / traj_std_flat
    history_t = torch.from_numpy(history_norm).unsqueeze(0).unsqueeze(-1).to(device)

    cond_mean = np.asarray(ckpt["global_cond_mean"], dtype=np.float32).squeeze(0)
    cond_std = np.asarray(ckpt["global_cond_std"], dtype=np.float32).squeeze(0)
    global_cond_norm = (global_cond - cond_mean) / cond_std
    global_cond_t = torch.from_numpy(global_cond_norm.astype(np.float32)).unsqueeze(0).to(device)

    total_steps = int(args.trajectory_length)
    history_k = int(ckpt["history_k"])
    num_generate = max(0, total_steps - history_k)

    generated = [history]
    history_window = history_t
    for _ in range(num_generate):
        next_step_norm = euler_sample(model, history_window, global_cond_t, num_steps=args.num_steps)
        next_step = (next_step_norm.squeeze(0).squeeze(-1).cpu().numpy() * traj_std_flat) + traj_mean_flat
        generated.append(next_step[None, :])
        next_step_t = torch.from_numpy((next_step - traj_mean_flat) / traj_std_flat)
        next_step_t = next_step_t.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
        history_window = torch.cat([history_window[:, 1:], next_step_t], dim=1)

    generated_arr = np.concatenate(generated, axis=0)

    sensor_names = ckpt.get("sensor_names")
    if sensor_names is None:
        sensor_names = [f"Concentration_{i+1}" for i in range(generated_arr.shape[1])]

    if time_values is None or len(time_values) != generated_arr.shape[0]:
        time_values = np.arange(generated_arr.shape[0], dtype=np.float32)

    out_df = pd.DataFrame(generated_arr, columns=sensor_names)
    out_df.insert(0, "Time", time_values[: generated_arr.shape[0]])

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Generated trajectory written to {args.output_csv}")
