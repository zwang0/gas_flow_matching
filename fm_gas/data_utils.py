import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectoryBatch:
    trajectories: np.ndarray
    time_values: np.ndarray
    sensor_names: List[str]
    positions: np.ndarray
    sensor_series: np.ndarray
    global_cond: np.ndarray
    inlet_ids: np.ndarray
    outlet_ids: np.ndarray
    sccm_values: np.ndarray


def _parse_metadata_from_name(file_name: str) -> Tuple[int, int, float]:
    match = re.search(r"Gas_3D_sim\d+_(\d+)_(\d+)_([\d.]+)sccm", file_name)
    if not match:
        raise ValueError(
            "Unable to parse inlet/outlet/sccm from filename. "
            f"Expected pattern Gas_3D_sim{{id}}_{{inlet}}_{{outlet}}_{{sccm}}sccm, got {file_name}."
        )
    inlet_id = int(match.group(1))
    outlet_id = int(match.group(2))
    sccm = float(match.group(3))
    return inlet_id, outlet_id, sccm


def _load_inlet_outlet_coords(coords_csv: str) -> dict[int, np.ndarray]:
    df = pd.read_csv(coords_csv)
    if not {"inlet_outlet", "x", "y", "z"}.issubset(set(df.columns)):
        raise ValueError(f"{coords_csv} must include columns inlet_outlet,x,y,z.")
    df["inlet_outlet"] = df["inlet_outlet"].astype(int)
    coords = {}
    for _, row in df.iterrows():
        coords[int(row["inlet_outlet"])] = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
    return coords


def _load_sensor_coords(sensor_coords_csv: str) -> np.ndarray:
    df = pd.read_csv(sensor_coords_csv)
    if not {"sensor", "x", "y", "z"}.issubset(set(df.columns)):
        raise ValueError(f"{sensor_coords_csv} must include columns sensor,x,y,z.")
    df = df.sort_values("sensor")
    return df[["x", "y", "z"]].to_numpy(dtype=np.float32)


def build_global_cond_from_files(
    binned_csv: str,
    surface_avg_csv: str,
    inlet_outlet_coords_csv: str,
    sensor_coords_csv: str,
) -> np.ndarray:
    inlet_id, outlet_id, sccm = _parse_metadata_from_name(os.path.basename(binned_csv))
    coords_map = _load_inlet_outlet_coords(inlet_outlet_coords_csv)
    if inlet_id not in coords_map or outlet_id not in coords_map:
        raise ValueError(
            "Inlet or outlet id not found in inlet_outlet_coords.csv. "
            f"Got inlet={inlet_id}, outlet={outlet_id} from {binned_csv}."
        )
    sensor_coords = _load_sensor_coords(sensor_coords_csv)
    sensor_times, sensor_series, _ = _load_surface_averages(surface_avg_csv)
    _, time_values, _ = _load_binned_csv(binned_csv)
    sensor_series = _align_sensor_series(sensor_times, sensor_series, time_values)
    inlet_xyz = coords_map[inlet_id]
    outlet_xyz = coords_map[outlet_id]
    sensor_coords_flat = sensor_coords.reshape(-1)
    sensor_series_flat = sensor_series.reshape(-1)
    return np.concatenate(
        [inlet_xyz, outlet_xyz, np.array([sccm], dtype=np.float32), sensor_coords_flat, sensor_series_flat], axis=0
    ).astype(np.float32)


def _parse_time_columns(columns: List[str]) -> Tuple[List[str], np.ndarray]:
    time_columns = [c for c in columns if c.startswith("c_")]
    if not time_columns:
        raise ValueError("No concentration columns found with prefix 'c_'.")
    times = np.array([float(c.split("_", 1)[1]) for c in time_columns], dtype=np.float32)
    return time_columns, times


def _load_binned_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"x_m", "y_m", "z_m"}.issubset(set(df.columns)):
        raise ValueError(f"{path} must include columns x_m,y_m,z_m.")
    time_columns, time_values = _parse_time_columns(list(df.columns))
    positions = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    traj_values = df[time_columns].to_numpy(dtype=np.float32)
    trajectories = traj_values.T
    return positions, time_values, trajectories


def _load_surface_averages(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Surface averages file {path} must include time and sensor columns.")
    time_col = df.columns[0]
    sensor_cols = list(df.columns[1:])
    time_values = df[time_col].to_numpy(dtype=np.float32)
    series = df[sensor_cols].to_numpy(dtype=np.float32)
    return time_values, series, sensor_cols


def _align_sensor_series(sensor_times: np.ndarray, sensor_series: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if np.allclose(sensor_times, target_times):
        return sensor_series
    aligned = np.zeros((len(target_times), sensor_series.shape[1]), dtype=np.float32)
    for i in range(sensor_series.shape[1]):
        aligned[:, i] = np.interp(target_times, sensor_times, sensor_series[:, i])
    return aligned


def load_trajectory_tensor(
    data_dir: str,
    inlet_outlet_coords_csv: str,
    sensor_coords_csv: str,
    max_files: int | None = None,
) -> TrajectoryBatch:
    binned_files = sorted(
        f
        for f in (os.path.join(data_dir, name) for name in os.listdir(data_dir))
        if f.endswith("_binned.csv")
    )
    if not binned_files:
        raise FileNotFoundError(
            f"No *_binned.csv files found in {data_dir}. "
            "Use --data-dir to point at the folder with binned trajectory files."
        )

    if max_files is not None:
        binned_files = binned_files[: max(1, int(max_files))]

    coords_map = _load_inlet_outlet_coords(inlet_outlet_coords_csv)
    sensor_coords = _load_sensor_coords(sensor_coords_csv)

    trajectories = []
    sensor_series_all = []
    inlet_ids = []
    outlet_ids = []
    sccm_values = []
    time_values_ref = None
    sensor_names_ref = None
    positions_ref = None

    for binned_path in binned_files:
        inlet_id, outlet_id, sccm = _parse_metadata_from_name(os.path.basename(binned_path))
        if inlet_id not in coords_map or outlet_id not in coords_map:
            raise ValueError(
                "Inlet or outlet id not found in inlet_outlet_coords.csv. "
                f"Got inlet={inlet_id}, outlet={outlet_id} from {binned_path}."
            )

        surface_path = binned_path.replace("_binned.csv", "_surface_averages.csv")
        if not os.path.exists(surface_path):
            raise FileNotFoundError(f"Surface averages file not found for {binned_path}.")

        positions, time_values, traj = _load_binned_csv(binned_path)
        sensor_times, sensor_series, sensor_names = _load_surface_averages(surface_path)
        sensor_series = _align_sensor_series(sensor_times, sensor_series, time_values)

        if positions_ref is None:
            positions_ref = positions
        elif not np.allclose(positions_ref, positions):
            raise ValueError("Binned positions are not identical across simulations.")

        if time_values_ref is None:
            time_values_ref = time_values
            sensor_names_ref = sensor_names
        else:
            if len(time_values) != len(time_values_ref):
                raise ValueError(
                    "All binned files must share the same time length. "
                    f"{binned_path} has {len(time_values)} vs {len(time_values_ref)}."
                )
            if sensor_names != sensor_names_ref:
                raise ValueError(
                    "All surface averages files must share the same sensor columns. "
                    f"{surface_path} has {sensor_names}."
                )

        trajectories.append(traj)
        sensor_series_all.append(sensor_series)
        inlet_ids.append(inlet_id)
        outlet_ids.append(outlet_id)
        sccm_values.append(sccm)

    traj_arr = np.stack(trajectories, axis=0).astype(np.float32)
    sensor_series_arr = np.stack(sensor_series_all, axis=0).astype(np.float32)
    inlet_ids_arr = np.asarray(inlet_ids, dtype=np.int64)
    outlet_ids_arr = np.asarray(outlet_ids, dtype=np.int64)
    sccm_arr = np.asarray(sccm_values, dtype=np.float32)

    inlet_xyz = np.stack([coords_map[i] for i in inlet_ids_arr], axis=0)
    outlet_xyz = np.stack([coords_map[i] for i in outlet_ids_arr], axis=0)
    sensor_coords_flat = sensor_coords.reshape(-1)
    sensor_series_flat = sensor_series_arr.reshape(sensor_series_arr.shape[0], -1)
    global_cond = np.concatenate(
        [
            inlet_xyz,
            outlet_xyz,
            sccm_arr[:, None],
            np.tile(sensor_coords_flat, (len(binned_files), 1)),
            sensor_series_flat,
        ],
        axis=1,
    ).astype(np.float32)

    return TrajectoryBatch(
        trajectories=traj_arr,
        time_values=time_values_ref,
        sensor_names=sensor_names_ref,
        positions=positions_ref,
        sensor_series=sensor_series_arr,
        global_cond=global_cond,
        inlet_ids=inlet_ids_arr,
        outlet_ids=outlet_ids_arr,
        sccm_values=sccm_arr,
    )


def normalize_trajectories(trajectories: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = trajectories.mean(axis=(0, 1), keepdims=True)
    std = trajectories.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, eps)
    return (trajectories - mean) / std, mean.astype(np.float32), std.astype(np.float32)


class HistoryWindowDataset(Dataset):
    def __init__(
        self,
        trajectories: torch.Tensor,
        global_cond: torch.Tensor,
        history_k: int,
    ):
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [num_traj, time, num_points].")
        self.trajectories = trajectories
        self.global_cond = global_cond
        self.history_k = int(history_k)
        if self.history_k < 1:
            raise ValueError("history_k must be >= 1.")
        self.num_traj, self.num_steps, _ = trajectories.shape
        if self.num_steps <= self.history_k:
            raise ValueError("Not enough timesteps to build history windows.")
        self.steps_per_traj = self.num_steps - self.history_k

    def __len__(self) -> int:
        return self.num_traj * self.steps_per_traj

    def __getitem__(self, idx: int):
        traj_idx = idx // self.steps_per_traj
        step_idx = idx % self.steps_per_traj
        t = step_idx + self.history_k
        field_history = self.trajectories[traj_idx, t - self.history_k : t]
        target = self.trajectories[traj_idx, t]
        cond = self.global_cond[traj_idx]
        return field_history, target.unsqueeze(-1), cond
