import torch
import torch.nn as nn
import torch.nn.functional as F


class TransEncoder(nn.Module):
    def __init__(self, time_points: int, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(time_points, hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.norm(x)


class AdaptiveHypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.v_to_e = nn.Linear(in_dim, out_dim)
        self.e_to_v = nn.Linear(out_dim, out_dim)
        self.att_v = nn.Linear(in_dim, 1)
        self.att_e = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = h.to(dtype=x.dtype)
        x_proj = self.v_to_e(x)
        deg_e = h.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        y0 = torch.bmm(h.transpose(1, 2), x_proj) / deg_e

        sv = self.att_v(x)
        se = self.att_e(y0).transpose(1, 2)
        w = torch.sigmoid(sv + se) * h

        w_v2e = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        y = torch.bmm(w_v2e.transpose(1, 2), x_proj)
        y = self.dropout(F.gelu(y))

        y_proj = self.e_to_v(y)
        w_e2v = w / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        out = torch.bmm(w_e2v, y_proj)
        out = self.dropout(F.gelu(out))
        return self.norm(out + self.residual(x))
