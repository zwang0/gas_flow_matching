from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponents = torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * exponents)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class SpatialSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, distance_bias: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + distance_bias
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        return self.out_proj(out)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SpatialSelfAttention(hidden_dim, num_heads)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, distance_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), distance_bias))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class AutoregressiveFlowMatcher(nn.Module):
    def __init__(
        self,
        sensor_positions: torch.Tensor,
        global_cond_dim: int = 7,
        history_k: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        if sensor_positions.ndim != 2 or sensor_positions.shape[1] != 3:
            raise ValueError("sensor_positions must have shape [num_nodes, 3].")
        self.history_k = int(history_k)
        self.num_nodes = int(sensor_positions.shape[0])
        self.global_cond_dim = int(global_cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        self.time_emb = SinusoidalTimeEmbedding(64)
        self.time_proj = nn.Sequential(nn.Linear(64, hidden_dim), nn.SiLU())
        self.input_proj = nn.Sequential(nn.Linear(self.history_k + 1, hidden_dim), nn.SiLU())
        self.global_proj = nn.Sequential(nn.Linear(self.global_cond_dim, hidden_dim), nn.SiLU())

        self.register_buffer("sensor_positions", sensor_positions, persistent=False)
        self.pos_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.dist_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_heads))
        self.register_buffer(
            "pairwise_dist",
            torch.cdist(sensor_positions, sensor_positions).unsqueeze(-1),
            persistent=False,
        )

        self.blocks = nn.ModuleList([SpatialTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        if history.shape[1] != self.history_k:
            raise ValueError("history has wrong length for history_k.")
        history_flat = history.squeeze(-1).transpose(1, 2)
        xt_flat = x_t.squeeze(-1)
        features = torch.cat([xt_flat.unsqueeze(-1), history_flat], dim=-1)
        hidden = self.input_proj(features)

        pos_emb = self.pos_mlp(self.sensor_positions).unsqueeze(0)
        dist_bias = self.dist_mlp(self.pairwise_dist).permute(2, 0, 1).unsqueeze(0)
        t_embed = self.time_proj(self.time_emb(t)).unsqueeze(1)
        global_embed = self.global_proj(global_cond).unsqueeze(1)
        hidden = hidden + pos_emb + t_embed + global_embed
        for block in self.blocks:
            hidden = block(hidden, dist_bias)
        return self.out_proj(self.out_norm(hidden))


def flow_matching_loss(
    model: AutoregressiveFlowMatcher,
    history: torch.Tensor,
    target_next: torch.Tensor,
    global_cond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = target_next.device
    batch_size = target_next.shape[0]
    x0 = torch.randn_like(target_next)
    t = torch.rand(batch_size, device=device)
    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * target_next
    target_v = target_next - x0
    pred_v = model(x_t, t, history, global_cond)
    loss = torch.mean((pred_v - target_v) ** 2)
    return loss, x_t, pred_v


@torch.no_grad()
def euler_sample(
    model: AutoregressiveFlowMatcher,
    history: torch.Tensor,
    global_cond: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    device = history.device
    batch_size, _, num_nodes, _ = history.shape
    x = torch.randn(batch_size, num_nodes, 1, device=device)
    dt = 1.0 / float(num_steps)
    for i in range(num_steps):
        t = torch.full((batch_size,), fill_value=i / num_steps, device=device)
        x = x + dt * model(x, t, history, global_cond)
    return x
