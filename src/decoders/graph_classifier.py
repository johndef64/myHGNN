"""
Graph classification decoder for batched heterogeneous graphs.

Pools node embeddings per graph and classifies at the graph level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add


class GraphClassifier(nn.Module):
    """
    Graph-level classifier with readout pooling + MLP.

    Pools node embeddings per graph using the batch assignment vector,
    then classifies each graph.
    """

    def __init__(self, hidden_dim, num_classes, dropout=0.3, pooling='mean', num_layers=2):
        """
        Args:
            hidden_dim: Dimension of node embeddings from encoder.
            num_classes: Number of graph classes.
            dropout: Dropout rate in MLP.
            pooling: Pooling strategy ('mean', 'max', 'sum', 'mean_max').
            num_layers: Number of MLP layers.
        """
        super().__init__()
        self.pooling = pooling

        pool_dim = hidden_dim * 2 if pooling == 'mean_max' else hidden_dim

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(pool_dim, num_classes))
        else:
            layers.append(nn.Linear(pool_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def pool(self, node_embeddings, batch):
        """
        Pool node embeddings per graph.

        Args:
            node_embeddings: [total_nodes, dim]
            batch: [total_nodes] LongTensor assigning each node to a graph index.

        Returns:
            graph_embeddings: [num_graphs, pool_dim]
        """
        if self.pooling == 'mean':
            return scatter_mean(node_embeddings, batch, dim=0)
        elif self.pooling == 'max':
            return scatter_max(node_embeddings, batch, dim=0)[0]
        elif self.pooling == 'sum':
            return scatter_add(node_embeddings, batch, dim=0)
        elif self.pooling == 'mean_max':
            mean_pool = scatter_mean(node_embeddings, batch, dim=0)
            max_pool = scatter_max(node_embeddings, batch, dim=0)[0]
            return torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, node_embeddings, rel_embeddings, batch):
        """
        Classify graphs given node embeddings and batch assignment.

        Args:
            node_embeddings: [total_nodes, dim] from encoder.
            rel_embeddings: [num_relations, dim] from encoder (unused, kept for interface).
            batch: [total_nodes] LongTensor assigning each node to its graph.

        Returns:
            logits: [num_graphs, num_classes]
        """
        graph_emb = self.pool(node_embeddings, batch)
        return self.mlp(graph_emb)

    def compute_loss(self, logits, labels):
        """Cross-entropy loss for graph classification."""
        return F.cross_entropy(logits, labels)
