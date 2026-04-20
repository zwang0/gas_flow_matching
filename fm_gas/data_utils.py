from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SensorPointMapping:
    indices: np.ndarray
    weights: np.ndarray
    distances: np.ndarray


@dataclass
class PreparedData:
    positions: np.ndarray
    trajectories: np.ndarray
    time_values: np.ndarray
    time_columns: List[str]
    sensor_series: np.ndarray
    sensor_coords: np.ndarray
    sensor_mapping: SensorPointMapping
    inlet_xyz: np.ndarray
    outlet_xyz: np.ndarray
    flow_sccm: float


def ensure_columns(df: pd.DataFrame, columns: List[str], file_path: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def load_inlet_outlet_xyz(coords_csv: str, inlet_id: int, outlet_id: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(coords_csv)
    ensure_columns(df, ["inlet_outlet", "x", "y", "z"], coords_csv)
    df["inlet_outlet"] = df["inlet_outlet"].astype(int)

    inlet_row = df[df["inlet_outlet"] == int(inlet_id)]
    outlet_row = df[df["inlet_outlet"] == int(outlet_id)]
    if inlet_row.empty:
        raise ValueError(f"Inlet id {inlet_id} not found in {coords_csv}")
    if outlet_row.empty:
        raise ValueError(f"Outlet id {outlet_id} not found in {coords_csv}")

    inlet_xyz = inlet_row[["x", "y", "z"]].iloc[0].to_numpy(dtype=np.float32)
    outlet_xyz = outlet_row[["x", "y", "z"]].iloc[0].to_numpy(dtype=np.float32)
    return inlet_xyz, outlet_xyz


def load_sensor_coords(sensor_coords_csv: str) -> np.ndarray:
    df = pd.read_csv(sensor_coords_csv)
    ensure_columns(df, ["sensor", "x", "y", "z"], sensor_coords_csv)
    df = df.sort_values("sensor")
    return df[["x", "y", "z"]].to_numpy(dtype=np.float32)


def parse_time_columns(columns: List[str]) -> Tuple[List[str], np.ndarray]:
    time_columns = [c for c in columns if c.startswith("c_")]
    if not time_columns:
        raise ValueError("No concentration columns found with prefix 'c_'.")
    times = np.array([float(c.split("_", 1)[1]) for c in time_columns], dtype=np.float32)
    return time_columns, times


def align_sensor_series(sensor_csv: str, target_times: np.ndarray) -> np.ndarray:
    sensor_df = pd.read_csv(sensor_csv)
    if sensor_df.shape[1] < 2:
        raise ValueError(f"Sensor file {sensor_csv} must contain time and sensor columns.")

    time_col = sensor_df.columns[0]
    sensor_cols = list(sensor_df.columns[1:])

    sensor_times = sensor_df[time_col].to_numpy(dtype=np.float32)
    sensor_matrix = sensor_df[sensor_cols].to_numpy(dtype=np.float32)

    sort_idx = np.argsort(sensor_times)
    sensor_times = sensor_times[sort_idx]
    sensor_matrix = sensor_matrix[sort_idx]

    aligned = np.zeros((len(target_times), sensor_matrix.shape[1]), dtype=np.float32)
    for i in range(sensor_matrix.shape[1]):
        aligned[:, i] = np.interp(target_times, sensor_times, sensor_matrix[:, i])
    return aligned


def build_sensor_point_mapping(positions: np.ndarray, sensor_coords: np.ndarray, k: int = 1) -> SensorPointMapping:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_points, 3].")
    if sensor_coords.ndim != 2 or sensor_coords.shape[1] != 3:
        raise ValueError("sensor_coords must have shape [num_sensors, 3].")
    if positions.shape[0] == 0:
        raise ValueError("positions is empty; cannot map sensors.")

    k = max(1, min(int(k), positions.shape[0]))

    # Distance matrix: [num_sensors, num_points]
    deltas = sensor_coords[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)

    topk_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    topk_dist = np.take_along_axis(distances, topk_idx, axis=1)

    order = np.argsort(topk_dist, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)
    topk_dist = np.take_along_axis(topk_dist, order, axis=1)

    if k == 1:
        weights = np.ones_like(topk_dist, dtype=np.float32)
    else:
        inv = 1.0 / np.maximum(topk_dist, 1e-6)
        weights = inv / np.sum(inv, axis=1, keepdims=True)

    if not np.all(np.isfinite(topk_dist)):
        raise ValueError("Non-finite sensor mapping distances detected.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite sensor mapping weights detected.")

    return SensorPointMapping(
        indices=topk_idx.astype(np.int64),
        weights=weights.astype(np.float32),
        distances=topk_dist.astype(np.float32),
    )


def prepare_data(
    traj_csv: str,
    sensor_csv: str,
    sensor_coords_csv: str,
    inlet_outlet_coords_csv: str,
    inlet_id: int,
    outlet_id: int,
    flow_sccm: float,
    sensor_map_k: int = 1,
) -> PreparedData:
    traj_df = pd.read_csv(traj_csv)
    ensure_columns(traj_df, ["x_m", "y_m", "z_m"], traj_csv)

    time_columns, times = parse_time_columns(list(traj_df.columns))
    positions = traj_df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    trajectories = traj_df[time_columns].to_numpy(dtype=np.float32)

    sensor_series = align_sensor_series(sensor_csv, times)
    sensor_coords = load_sensor_coords(sensor_coords_csv)

    if sensor_series.shape[1] != sensor_coords.shape[0]:
        raise ValueError(
            "Number of sensor series columns does not match number of sensor coordinates. "
            f"Series sensors={sensor_series.shape[1]}, coords sensors={sensor_coords.shape[0]}."
        )

    inlet_xyz, outlet_xyz = load_inlet_outlet_xyz(inlet_outlet_coords_csv, inlet_id, outlet_id)
    sensor_mapping = build_sensor_point_mapping(positions=positions, sensor_coords=sensor_coords, k=sensor_map_k)

    return PreparedData(
        positions=positions,
        trajectories=trajectories,
        time_values=times,
        time_columns=time_columns,
        sensor_series=sensor_series,
        sensor_coords=sensor_coords,
        sensor_mapping=sensor_mapping,
        inlet_xyz=inlet_xyz,
        outlet_xyz=outlet_xyz,
        flow_sccm=float(flow_sccm),
    )
