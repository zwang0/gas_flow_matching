import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data_utils import prepare_data
from .features import (
    build_global_condition,
    build_sensor_condition_matrix,
    make_condition_matrix,
    normalize_features,
)
from .model import ConditionalFlowMatcher, TrajectoryDataset


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constraint_mode = str(args.constraint_mode).lower()

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

    traj_norm, traj_mean, traj_std = normalize_features(prepared.trajectories)
    pos_norm, pos_mean, pos_std = normalize_features(prepared.positions)
    global_cond = build_global_condition(prepared)
    cond_matrix = make_condition_matrix(pos_norm, global_cond)

    sensor_targets = prepared.sensor_series.T.astype(np.float32)
    sensor_targets_norm = ((sensor_targets - traj_mean) / traj_std).astype(np.float32)
    sensor_cond_matrix = build_sensor_condition_matrix(
        cond_matrix=cond_matrix,
        sensor_indices=prepared.sensor_mapping.indices,
        sensor_weights=prepared.sensor_mapping.weights,
    )

    sensor_targets_t = torch.from_numpy(sensor_targets_norm).to(device)
    sensor_cond_t = torch.from_numpy(sensor_cond_matrix).to(device)

    dataset = TrajectoryDataset(torch.from_numpy(traj_norm), torch.from_numpy(cond_matrix))
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = ConditionalFlowMatcher(
        traj_dim=traj_norm.shape[1],
        cond_dim=cond_matrix.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if args.lambda_sensor_warmup_epochs > 0:
            warmup = min(1.0, float(epoch) / float(args.lambda_sensor_warmup_epochs))
        else:
            warmup = 1.0
        lambda_eff = float(args.lambda_sensor) * warmup

        model.train()
        train_fm_losses = []
        train_sensor_losses = []
        for x1, cond in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            x1 = x1.to(device)
            cond = cond.to(device)

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            target_v = x1 - x0

            fm_loss = torch.mean((model(x_t, t, cond) - target_v) ** 2)

            sensor_loss = torch.tensor(0.0, device=device)
            if constraint_mode in {"soft", "hybrid"}:
                x0_s = torch.randn_like(sensor_targets_t)
                t_s = torch.rand(sensor_targets_t.shape[0], device=device)
                x_t_s = (1.0 - t_s[:, None]) * x0_s + t_s[:, None] * sensor_targets_t
                pred_v_s = model(x_t_s, t_s, sensor_cond_t)
                # Recover endpoint estimate x1 from linear bridge relation.
                x1_hat_s = x_t_s + (1.0 - t_s[:, None]) * pred_v_s
                sensor_loss = torch.mean((x1_hat_s - sensor_targets_t) ** 2)

            loss = fm_loss + (lambda_eff * sensor_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_fm_losses.append(fm_loss.item())
            train_sensor_losses.append(sensor_loss.item())

        model.eval()
        val_fm_losses = []
        val_sensor_losses = []
        with torch.no_grad():
            for x1, cond in val_loader:
                x1 = x1.to(device)
                cond = cond.to(device)
                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], device=device)
                x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
                target_v = x1 - x0
                fm_loss = torch.mean((model(x_t, t, cond) - target_v) ** 2)

                sensor_loss = torch.tensor(0.0, device=device)
                if constraint_mode in {"soft", "hybrid"}:
                    x0_s = torch.randn_like(sensor_targets_t)
                    t_s = torch.rand(sensor_targets_t.shape[0], device=device)
                    x_t_s = (1.0 - t_s[:, None]) * x0_s + t_s[:, None] * sensor_targets_t
                    pred_v_s = model(x_t_s, t_s, sensor_cond_t)
                    x1_hat_s = x_t_s + (1.0 - t_s[:, None]) * pred_v_s
                    sensor_loss = torch.mean((x1_hat_s - sensor_targets_t) ** 2)

                val_fm_losses.append(fm_loss.item())
                val_sensor_losses.append(sensor_loss.item())

        train_fm = float(np.mean(train_fm_losses)) if train_fm_losses else float("nan")
        train_sensor = float(np.mean(train_sensor_losses)) if train_sensor_losses else float("nan")
        val_fm = float(np.mean(val_fm_losses)) if val_fm_losses else float("nan")
        val_sensor = float(np.mean(val_sensor_losses)) if val_sensor_losses else float("nan")

        train_loss = train_fm + (lambda_eff * train_sensor)
        val_loss = val_fm + (lambda_eff * val_sensor)
        print(
            f"Epoch {epoch:04d} | train_fm={train_fm:.6f} | train_sensor={train_sensor:.6f} "
            f"| val_fm={val_fm:.6f} | val_sensor={val_sensor:.6f} | lambda_sensor={lambda_eff:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "traj_dim": traj_norm.shape[1],
                    "cond_dim": cond_matrix.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "time_columns": prepared.time_columns,
                    "traj_mean": traj_mean,
                    "traj_std": traj_std,
                    "pos_mean": pos_mean,
                    "pos_std": pos_std,
                    "constraint_mode": constraint_mode,
                    "lambda_sensor": float(args.lambda_sensor),
                    "lambda_sensor_warmup_epochs": int(args.lambda_sensor_warmup_epochs),
                    "sensor_map_k": int(args.sensor_map_k),
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")
