import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, conditions: torch.Tensor):
        self.trajectories = trajectories
        self.conditions = conditions

    def __len__(self) -> int:
        return self.trajectories.shape[0]

    def __getitem__(self, idx: int):
        return self.trajectories[idx], self.conditions[idx]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponents = torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        freqs = torch.exp(-np.log(10000.0) * exponents)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def _choose_num_heads(hidden_dim: int, max_heads: int = 8) -> int:
    for num_heads in range(min(max_heads, hidden_dim), 0, -1):
        if hidden_dim % num_heads == 0:
            return num_heads
    return 1


class ConditionalFlowMatcher(nn.Module):
    def __init__(
        self,
        traj_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int | None = None,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.time_emb = SinusoidalTimeEmbedding(64)
        num_heads = num_heads or _choose_num_heads(hidden_dim)

        self.traj_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU())
        self.time_proj = nn.Sequential(nn.Linear(64, hidden_dim), nn.SiLU())
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU())
        self.pos_emb = nn.Parameter(torch.zeros(1, traj_dim, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        xt_tokens = self.traj_proj(x_t.unsqueeze(-1))
        t_h = self.time_proj(self.time_emb(t)).unsqueeze(1)
        c_h = self.cond_proj(cond).unsqueeze(1)
        hidden = xt_tokens + self.pos_emb + t_h + c_h
        hidden = self.transformer(hidden)
        return self.output_proj(self.output_norm(hidden)).squeeze(-1)


def sample_trajectories(
    model: ConditionalFlowMatcher,
    cond: torch.Tensor,
    traj_dim: int,
    num_steps: int,
    device: torch.device,
    constraint_mode: str = "none",
    sensor_targets: torch.Tensor | None = None,
    sensor_indices: torch.Tensor | None = None,
    sensor_weights: torch.Tensor | None = None,
    projection_alpha: float = 1.0,
    projection_every: int = 1,
) -> torch.Tensor:
    def project_sensor_constraints(x: torch.Tensor) -> torch.Tensor:
        if sensor_targets is None or sensor_indices is None or sensor_weights is None:
            return x

        alpha = float(np.clip(projection_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return x

        k = sensor_indices.shape[1]
        if k == 1:
            idx = sensor_indices[:, 0]
            current = x[idx]
            x[idx] = (1.0 - alpha) * current + alpha * sensor_targets
            return x

        # For k-NN mappings, distribute sensor correction back to mapped points.
        for s in range(sensor_indices.shape[0]):
            idx_s = sensor_indices[s]
            w_s = sensor_weights[s]
            pred_s = torch.sum(w_s[:, None] * x[idx_s], dim=0)
            delta = sensor_targets[s] - pred_s
            for j in range(k):
                x[idx_s[j]] = x[idx_s[j]] + alpha * w_s[j] * delta
        return x

    x = torch.randn(cond.shape[0], traj_dim, device=device)
    dt = 1.0 / float(num_steps)
    projection_every = max(1, int(projection_every))
    do_projection = constraint_mode in {"hard", "hybrid"}

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((cond.shape[0],), fill_value=i / num_steps, device=device)
            x = x + dt * model(x, t, cond)
            if do_projection and ((i + 1) % projection_every == 0):
                x = project_sensor_constraints(x)
    return x
