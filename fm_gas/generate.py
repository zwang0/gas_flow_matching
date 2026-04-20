import os

import numpy as np
import pandas as pd
import torch

from .data_utils import build_sensor_point_mapping, ensure_columns, prepare_data
from .features import build_global_condition, make_condition_matrix
from .model import ConditionalFlowMatcher, sample_trajectories


def generate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constraint_mode = str(args.constraint_mode).lower()
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ConditionalFlowMatcher(
        traj_dim=int(ckpt["traj_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prepared = prepare_data(
        traj_csv=args.traj_csv,
        sensor_csv=args.sensor_csv,
        sensor_coords_csv=args.sensor_coords_csv,
        inlet_outlet_coords_csv=args.inlet_outlet_coords_csv,
        inlet_id=args.inlet_id,
        outlet_id=args.outlet_id,
        flow_sccm=args.flow_sccm,
        sensor_map_k=args.sensor_map_k,
    )

    if args.positions_csv:
        pos_df = pd.read_csv(args.positions_csv)
        ensure_columns(pos_df, ["x_m", "y_m", "z_m"], args.positions_csv)
        positions = pos_df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    else:
        positions = prepared.positions

    pos_mean = np.asarray(ckpt["pos_mean"], dtype=np.float32)
    pos_std = np.asarray(ckpt["pos_std"], dtype=np.float32)
    pos_norm = (positions - pos_mean) / pos_std

    cond_matrix = make_condition_matrix(pos_norm, build_global_condition(prepared))
    cond_tensor = torch.from_numpy(cond_matrix).to(device)

    sensor_mapping = build_sensor_point_mapping(
        positions=positions,
        sensor_coords=prepared.sensor_coords,
        k=args.sensor_map_k,
    )

    traj_mean = np.asarray(ckpt["traj_mean"], dtype=np.float32)
    traj_std = np.asarray(ckpt["traj_std"], dtype=np.float32)
    sensor_targets_norm = ((prepared.sensor_series.T.astype(np.float32) - traj_mean) / traj_std).astype(np.float32)

    mode_for_sampling = constraint_mode
    if positions.shape[0] < prepared.sensor_coords.shape[0] and constraint_mode in {"hard", "hybrid"}:
        print(
            "Warning: fewer generated positions than sensors; disabling hard projection for this run. "
            "Use full-field generation to apply hard/hybrid sensor projection."
        )
        mode_for_sampling = "none"

    x_norm = sample_trajectories(
        model=model,
        cond=cond_tensor,
        traj_dim=int(ckpt["traj_dim"]),
        num_steps=args.num_steps,
        device=device,
        constraint_mode=mode_for_sampling,
        sensor_targets=torch.from_numpy(sensor_targets_norm).to(device),
        sensor_indices=torch.from_numpy(sensor_mapping.indices).to(device),
        sensor_weights=torch.from_numpy(sensor_mapping.weights).to(device),
        projection_alpha=args.projection_alpha,
        projection_every=args.projection_every,
    ).cpu().numpy()

    generated = x_norm * traj_std + traj_mean

    out_df = pd.DataFrame(positions, columns=["x_m", "y_m", "z_m"])
    for i, c in enumerate(ckpt["time_columns"]):
        out_df[c] = generated[:, i]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Generated trajectories written to {args.output_csv}")
