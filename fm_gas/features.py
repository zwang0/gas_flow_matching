from typing import Tuple

import numpy as np

from .data_utils import PreparedData


def build_global_condition(prepared: PreparedData) -> np.ndarray:
    sensor_flat = prepared.sensor_series.reshape(-1)
    sensor_coords_flat = prepared.sensor_coords.reshape(-1)
    cond = np.concatenate(
        [
            sensor_flat,
            sensor_coords_flat,
            prepared.inlet_xyz,
            prepared.outlet_xyz,
            np.array([prepared.flow_sccm], dtype=np.float32),
        ]
    ).astype(np.float32)
    return cond


def normalize_features(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (arr - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def make_condition_matrix(pos_norm: np.ndarray, global_cond: np.ndarray) -> np.ndarray:
    global_tiled = np.repeat(global_cond[None, :], repeats=pos_norm.shape[0], axis=0)
    return np.concatenate([pos_norm, global_tiled], axis=1).astype(np.float32)


def build_sensor_condition_matrix(
    cond_matrix: np.ndarray,
    sensor_indices: np.ndarray,
    sensor_weights: np.ndarray,
) -> np.ndarray:
    sensor_cond_points = cond_matrix[sensor_indices]
    weighted = sensor_cond_points * sensor_weights[..., None]
    return np.sum(weighted, axis=1).astype(np.float32)


def extract_mapped_trajectories(
    trajectories: np.ndarray,
    sensor_indices: np.ndarray,
    sensor_weights: np.ndarray,
) -> np.ndarray:
    mapped_points = trajectories[sensor_indices]
    weighted = mapped_points * sensor_weights[..., None]
    return np.sum(weighted, axis=1).astype(np.float32)
