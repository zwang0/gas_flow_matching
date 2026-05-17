import os

import numpy as np
import pandas as pd


def load_sensor_positions(path: str) -> np.ndarray:
    if not path:
        raise ValueError("sensor positions path must be provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sensor positions file not found: {path}. "
            "Provide --sensor-coords-csv with columns sensor,x,y,z."
        )
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if cols[:4] == ["sensor", "x", "y", "z"]:
        df = df.sort_values("sensor")
    if cols[:3] != ["x", "y", "z"]:
        df.columns = [c.lower() for c in df.columns]
    if not {"x", "y", "z"}.issubset(set(df.columns)):
        raise ValueError(f"{path} must include columns sensor,x,y,z or x,y,z.")
    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if positions.shape[0] != 12:
        raise ValueError(
            f"Expected 12 sensor positions, got {positions.shape[0]}. "
            "Update the CSV or change the model to match your sensor count."
        )
    return positions


def default_sensor_positions_path(data_dir: str) -> str:
    return os.path.join(data_dir, "sensor_coords.csv")
