"""Spatiotemporal GraphSAGE model used by ARTON.

The module uses PyTorch Geometric when installed. A small, pure-PyTorch SAGEConv
fallback is included so the simulator and component tests can run on machines where
PyG wheels are unavailable (including CPU-only demo environments).
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional dependency; the fallback keeps simulation hardware-independent.
    from torch_geometric.nn import SAGEConv, global_mean_pool  # type: ignore
    HAS_TORCH_GEOMETRIC = True
except ImportError:  # pragma: no cover - exercised on installations without PyG
    HAS_TORCH_GEOMETRIC = False

    class SAGEConv(nn.Module):
        """Mean-neighbour GraphSAGE convolution implemented with torch ops."""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_root = nn.Linear(in_channels, out_channels, bias=True)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            n = int(x.shape[0])
            neighbour_sum = torch.zeros_like(x)
            degree = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
            if edge_index.numel():
                src, dst = edge_index[0].long(), edge_index[1].long()
                neighbour_sum.index_add_(0, dst, x[src])
                degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device))
            neighbour_mean = neighbour_sum / degree.clamp_min(1.0)
            return self.lin_neigh(neighbour_mean) + self.lin_root(x)

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return x.new_zeros((0, x.shape[-1]))
        count = int(batch.max().item()) + 1
        out = x.new_zeros((count, x.shape[-1]))
        out.index_add_(0, batch.long(), x)
        denom = torch.bincount(batch.long(), minlength=count).to(x.dtype).clamp_min(1).unsqueeze(1)
        return out / denom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatiotemporalGraphSAGE(nn.Module):
    """GraphSAGE stack with a multi-step delay prediction head."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        self.convs.extend(SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1))
        self.linear1 = nn.Linear(hidden_dim, max(1, hidden_dim // 2))
        self.linear2 = nn.Linear(max(1, hidden_dim // 2), output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        logger.info("Initialized GraphSAGE %s->%s->%s (PyG=%s)", input_dim, hidden_dim, output_dim, HAS_TORCH_GEOMETRIC)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor | None = None):
        x = x.float()
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        node_embeddings = x
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = torch.mean(x, dim=0, keepdim=True)
        predictions = self.linear2(self.dropout_layer(F.relu(self.linear1(node_embeddings))))
        return node_embeddings, graph_embedding, predictions

    def predict_delay(self, graph_data: Any, node_features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return [num_nodes, output_dim] predictions as a CPU NumPy array."""
        if isinstance(graph_data, nx.Graph):
            nodes = list(graph_data.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            edges: list[tuple[int, int]] = []
            for u, v in graph_data.edges():
                ui, vi = node_to_idx[u], node_to_idx[v]
                edges.extend(((ui, vi), (vi, ui)))  # GraphSAGE expects both directions.
            edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                          if edges else torch.empty((2, 0), dtype=torch.long))
        elif hasattr(graph_data, "edge_index"):
            edge_index = graph_data.edge_index
        else:
            edge_index = graph_data[1]
        device = next(self.parameters()).device
        features = node_features if isinstance(node_features, torch.Tensor) else torch.as_tensor(node_features)
        features = features.to(device=device, dtype=torch.float32)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"node_features must have shape [N, {self.input_dim}], got {tuple(features.shape)}")
        self.eval()
        with torch.no_grad():
            _, _, delay_preds = self.forward(features, edge_index)
        return delay_preds.detach().cpu().numpy()


def create_railway_graph_from_schedule(schedule_data=None) -> nx.Graph:
    """Create a small connected demo topology; replace with real track data later."""
    return nx.grid_graph(dim=[5, 4])


def extract_spatiotemporal_features(schedule_data=None, current_time=None, horizon: int = 5,
                                    num_nodes: int = 20, seed: int | None = None) -> torch.Tensor:
    """Generate clearly synthetic node features for a hardware-free demonstration."""
    rng = np.random.default_rng(seed)
    now = float(current_time if current_time is not None else 0.0)
    feats = rng.normal(0, 0.15, size=(num_nodes, 10)).astype(np.float32)
    feats[:, 0] = np.linspace(0.0, 1.0, num_nodes)  # normalized node index
    feats[:, 1] = np.sin(now / 60.0)
    feats[:, 2] = np.cos(now / 60.0)
    feats[:, 3] = np.sin(now / 3600.0)
    feats[:, 4] = np.cos(now / 3600.0)
    feats[:, 5] = rng.uniform(0, 1, num_nodes)  # synthetic occupancy proxy
    return torch.from_numpy(feats)
