import os

import numpy as np
import pandas as pd
import torch

from .data_utils import build_global_cond_from_files
from .model import AutoregressiveFlowMatcher, PrefixInitializer, euler_sample


def _load_binned_positions_and_time(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"x_m", "y_m", "z_m"}.issubset(set(df.columns)):
        raise ValueError(f"{path} must include columns x_m,y_m,z_m.")
    time_columns = [c for c in df.columns if c.startswith("c_")]
    if not time_columns:
        raise ValueError("No concentration columns found with prefix 'c_'.")
    time_values = np.array([float(c.split("_", 1)[1]) for c in time_columns], dtype=np.float32)
    positions = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    traj_values = df[time_columns].to_numpy(dtype=np.float32)
    trajectories = traj_values.T
    return positions, time_values, trajectories


def _load_surface_averages(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Surface averages file {path} must include time and sensor columns.")
    time_values = df.iloc[:, 0].to_numpy(dtype=np.float32)
    series = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return time_values, series


def _align_sensor_series(sensor_times: np.ndarray, sensor_series: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if np.allclose(sensor_times, target_times):
        return sensor_series
    aligned = np.zeros((len(target_times), sensor_series.shape[1]), dtype=np.float32)
    for i in range(sensor_series.shape[1]):
        aligned[:, i] = np.interp(target_times, sensor_times, sensor_series[:, i])
    return aligned


def _field_history_window(series: np.ndarray, t: int, history_k: int) -> np.ndarray:
    start = max(0, t - history_k)
    window = series[start:t]
    if window.shape[0] < history_k:
        seed = window[:1] if window.shape[0] > 0 else series[:1]
        pad = np.repeat(seed, repeats=history_k - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window


def generate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    positions, time_values, trajectories = _load_binned_positions_and_time(args.binned_csv)
    sensor_times, sensor_series = _load_surface_averages(args.surface_avg_csv)
    sensor_series = _align_sensor_series(sensor_times, sensor_series, time_values)

    model = AutoregressiveFlowMatcher(
        point_positions=torch.from_numpy(positions).to(device),
        global_cond_dim=int(ckpt["global_cond_mean"].shape[-1]),
        history_k=int(ckpt["history_k"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        num_heads=int(ckpt["num_heads"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prefix_model = PrefixInitializer(
        point_positions=torch.from_numpy(positions).to(device),
        global_cond_dim=int(ckpt["global_cond_mean"].shape[-1]),
        history_k=int(ckpt["history_k"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(device)
    if "prefix_state_dict" in ckpt:
        prefix_model.load_state_dict(ckpt["prefix_state_dict"])
    prefix_model.eval()

    global_cond = build_global_cond_from_files(
        args.binned_csv,
        args.surface_avg_csv,
        args.inlet_outlet_coords_csv,
        args.sensor_coords_csv,
    )

    traj_mean = np.asarray(ckpt["traj_mean"], dtype=np.float32)
    traj_std = np.asarray(ckpt["traj_std"], dtype=np.float32)
    traj_mean_flat = traj_mean.squeeze(0).squeeze(0)
    traj_std_flat = traj_std.squeeze(0).squeeze(0)

    cond_mean = np.asarray(ckpt["global_cond_mean"], dtype=np.float32).squeeze(0)
    cond_std = np.asarray(ckpt["global_cond_std"], dtype=np.float32).squeeze(0)
    global_cond_norm = (global_cond - cond_mean) / cond_std
    global_cond_t = torch.from_numpy(global_cond_norm.astype(np.float32)).unsqueeze(0).to(device)

    history_k = int(ckpt["history_k"])
    if args.cold_start:
        if "prefix_state_dict" not in ckpt:
            raise ValueError("Checkpoint missing prefix_state_dict; retrain with prefix enabled.")
        prefix_norm = prefix_model(global_cond_t).squeeze(0).squeeze(-1)
        generated_norm = [prefix_norm[i] for i in range(history_k)]
    else:
        trajectories_norm = (trajectories - traj_mean_flat[None, :]) / traj_std_flat[None, :]
        generated_norm = [trajectories_norm[i] for i in range(history_k)]

    for t_idx in range(history_k, len(time_values)):
        field_hist = _field_history_window(np.stack(generated_norm, axis=0), t_idx, history_k)
        field_hist_t = torch.from_numpy(field_hist).unsqueeze(0).to(device)
        next_step_norm = euler_sample(model, field_hist_t, global_cond_t, num_steps=args.num_steps)
        next_step = next_step_norm.squeeze(0).squeeze(-1).cpu().numpy()
        generated_norm.append(next_step)

    generated_arr = (np.stack(generated_norm, axis=1) * traj_std_flat[:, None]) + traj_mean_flat[:, None]
    out_df = pd.DataFrame(positions, columns=["x_m", "y_m", "z_m"])
    for idx, t_val in enumerate(time_values):
        out_df[f"c_{t_val:.4f}"] = generated_arr[:, idx]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Generated trajectory written to {args.output_csv}")
