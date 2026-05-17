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
    global_cond: np.ndarray
    inlet_ids: np.ndarray
    outlet_ids: np.ndarray
    sccm_values: np.ndarray


def _load_surface_averages(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Surface averages file {path} must include time and sensor columns.")
    time_col = df.columns[0]
    sensor_cols = list(df.columns[1:])
    time_values = df[time_col].to_numpy(dtype=np.float32)
    series = df[sensor_cols].to_numpy(dtype=np.float32)
    return time_values, series, sensor_cols


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


def build_global_cond_from_filename(file_path: str, inlet_outlet_coords_csv: str) -> np.ndarray:
    inlet_id, outlet_id, sccm = _parse_metadata_from_name(os.path.basename(file_path))
    coords_map = _load_inlet_outlet_coords(inlet_outlet_coords_csv)
    if inlet_id not in coords_map or outlet_id not in coords_map:
        raise ValueError(
            "Inlet or outlet id not found in inlet_outlet_coords.csv. "
            f"Got inlet={inlet_id}, outlet={outlet_id} from {file_path}."
        )
    inlet_xyz = coords_map[inlet_id]
    outlet_xyz = coords_map[outlet_id]
    return np.concatenate([inlet_xyz, outlet_xyz, np.array([sccm], dtype=np.float32)], axis=0)


def load_trajectory_tensor(
    data_dir: str,
    inlet_outlet_coords_csv: str,
    max_files: int | None = None,
) -> TrajectoryBatch:
    files = sorted(
        f
        for f in (os.path.join(data_dir, name) for name in os.listdir(data_dir))
        if f.endswith("_surface_averages.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No *_surface_averages.csv files found in {data_dir}. "
            "Use --data-dir to point at the folder with surface averages files."
        )

    if max_files is not None:
        files = files[: max(1, int(max_files))]

    trajectories = []
    inlet_ids = []
    outlet_ids = []
    sccm_values = []
    time_values_ref = None
    sensor_names_ref = None
    coords_map = _load_inlet_outlet_coords(inlet_outlet_coords_csv)
    for path in files:
        inlet_id, outlet_id, sccm = _parse_metadata_from_name(os.path.basename(path))
        if inlet_id not in coords_map or outlet_id not in coords_map:
            raise ValueError(
                "Inlet or outlet id not found in inlet_outlet_coords.csv. "
                f"Got inlet={inlet_id}, outlet={outlet_id} from {path}."
            )
        time_values, series, sensor_names = _load_surface_averages(path)
        if time_values_ref is None:
            time_values_ref = time_values
            sensor_names_ref = sensor_names
        else:
            if len(time_values) != len(time_values_ref):
                raise ValueError(
                    "All surface averages files must share the same time length. "
                    f"{path} has {len(time_values)} vs {len(time_values_ref)}."
                )
            if sensor_names != sensor_names_ref:
                raise ValueError(
                    "All surface averages files must share the same sensor columns. "
                    f"{path} has {sensor_names}."
                )
        trajectories.append(series)
        inlet_ids.append(inlet_id)
        outlet_ids.append(outlet_id)
        sccm_values.append(sccm)

    traj_arr = np.stack(trajectories, axis=0).astype(np.float32)
    inlet_ids_arr = np.asarray(inlet_ids, dtype=np.int64)
    outlet_ids_arr = np.asarray(outlet_ids, dtype=np.int64)
    sccm_arr = np.asarray(sccm_values, dtype=np.float32)

    inlet_xyz = np.stack([coords_map[i] for i in inlet_ids_arr], axis=0)
    outlet_xyz = np.stack([coords_map[i] for i in outlet_ids_arr], axis=0)
    global_cond = np.concatenate([inlet_xyz, outlet_xyz, sccm_arr[:, None]], axis=1).astype(np.float32)

    return TrajectoryBatch(
        trajectories=traj_arr,
        time_values=time_values_ref,
        sensor_names=sensor_names_ref,
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
    def __init__(self, trajectories: torch.Tensor, global_cond: torch.Tensor, history_k: int):
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [num_traj, time, num_nodes].")
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
        history = self.trajectories[traj_idx, t - self.history_k : t]
        target = self.trajectories[traj_idx, t]
        cond = self.global_cond[traj_idx]
        return history.unsqueeze(-1), target.unsqueeze(-1), cond
