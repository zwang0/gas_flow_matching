import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .data_utils import HistoryWindowDataset, load_trajectory_tensor, normalize_trajectories
from .model import AutoregressiveFlowMatcher, PrefixInitializer, flow_matching_loss


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_traj = None if int(args.max_traj) <= 0 else int(args.max_traj)
    batch = load_trajectory_tensor(
        args.data_dir,
        inlet_outlet_coords_csv=args.inlet_outlet_coords_csv,
        sensor_coords_csv=args.sensor_coords_csv,
        max_files=max_traj,
    )
    trajectories, traj_mean, traj_std = normalize_trajectories(batch.trajectories)


    cond_mean = batch.global_cond.mean(axis=0, keepdims=True)
    cond_std = batch.global_cond.std(axis=0, keepdims=True)
    cond_std = np.maximum(cond_std, 1e-6)
    global_cond_norm = (batch.global_cond - cond_mean) / cond_std

    dataset = HistoryWindowDataset(
        torch.from_numpy(trajectories),
        torch.from_numpy(global_cond_norm.astype(np.float32)),
        history_k=args.history_k,
    )
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = AutoregressiveFlowMatcher(
        point_positions=torch.from_numpy(batch.positions).to(device),
        global_cond_dim=global_cond_norm.shape[1],
        history_k=args.history_k,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    prefix_model = PrefixInitializer(
        point_positions=torch.from_numpy(batch.positions).to(device),
        global_cond_dim=global_cond_norm.shape[1],
        history_k=args.history_k,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(prefix_model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    best_val = float("inf")
    save_every = int(args.save_every) if hasattr(args, "save_every") else 0
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    ckpt_root, ckpt_ext = os.path.splitext(args.checkpoint)
    if not ckpt_ext:
        ckpt_ext = ".pt"

    prefix_targets = torch.from_numpy(trajectories[:, : args.history_k]).to(device)
    prefix_cond = torch.from_numpy(global_cond_norm.astype(np.float32)).to(device)
    prefix_dataset = TensorDataset(prefix_cond, prefix_targets)
    prefix_batch_size = args.prefix_batch_size if int(args.prefix_batch_size) > 0 else args.batch_size
    prefix_loader = DataLoader(prefix_dataset, batch_size=prefix_batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, args.epochs + 1):
        prefix_model.train()
        prefix_losses = []
        for cond_batch, prefix_batch in prefix_loader:
            pred_prefix = prefix_model(cond_batch)
            loss_prefix = torch.mean((pred_prefix - prefix_batch.unsqueeze(-1)) ** 2)
            optimizer.zero_grad()
            (float(args.prefix_loss_weight) * loss_prefix).backward()
            optimizer.step()
            prefix_losses.append(loss_prefix.item())

        model.train()
        train_losses = []
        for field_history, target, global_cond in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False
        ):
            field_history = field_history.to(device)
            target = target.to(device)
            global_cond = global_cond.to(device)
            loss, _, _ = flow_matching_loss(model, field_history, target, global_cond)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for field_history, target, global_cond in val_loader:
                field_history = field_history.to(device)
                target = target.to(device)
                global_cond = global_cond.to(device)
                loss, _, _ = flow_matching_loss(model, field_history, target, global_cond)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        prefix_loss = float(np.mean(prefix_losses)) if prefix_losses else float("nan")
        print(
            f"Epoch {epoch:04d} | prefix_loss={prefix_loss:.6f} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}"
        )

        if save_every > 0 and (epoch % save_every == 0):
            epoch_path = f"{ckpt_root}_epoch{epoch:04d}{ckpt_ext}"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "prefix_state_dict": prefix_model.state_dict(),
                    "history_k": int(args.history_k),
                    "num_nodes": int(batch.trajectories.shape[2]),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "num_heads": int(args.num_heads),
                    "traj_mean": traj_mean,
                    "traj_std": traj_std,
                    "time_values": batch.time_values,
                    "sensor_names": batch.sensor_names,
                    "point_positions": batch.positions.astype(np.float32),
                    "global_cond_mean": cond_mean.astype(np.float32),
                    "global_cond_std": cond_std.astype(np.float32),
                },
                epoch_path,
            )
            print(f"Saved checkpoint to {epoch_path}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "prefix_state_dict": prefix_model.state_dict(),
                    "history_k": int(args.history_k),
                    "num_nodes": int(batch.trajectories.shape[2]),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "num_heads": int(args.num_heads),
                    "traj_mean": traj_mean,
                    "traj_std": traj_std,
                    "time_values": batch.time_values,
                    "sensor_names": batch.sensor_names,
                    "point_positions": batch.positions.astype(np.float32),
                    "global_cond_mean": cond_mean.astype(np.float32),
                    "global_cond_std": cond_std.astype(np.float32),
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")
