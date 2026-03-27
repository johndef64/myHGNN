"""
Node classification decoder for heterogeneous graphs.

Takes node embeddings from an encoder and classifies a subset of labeled nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeClassifier(nn.Module):
    """
    MLP-based node classifier.

    Takes node embeddings from a GNN encoder and produces class logits
    for labeled nodes.
    """

    def __init__(self, hidden_dim, num_classes, dropout=0.3, num_layers=2):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_embeddings, rel_embeddings, node_indices):
        """
        Classify nodes at given indices.

        Args:
            node_embeddings: [num_entities, dim] from encoder.
            rel_embeddings: [num_relations, dim] from encoder (unused, kept for interface).
            node_indices: [num_labeled] LongTensor of node IDs to classify.

        Returns:
            logits: [num_labeled, num_classes]
        """
        x = node_embeddings[node_indices]
        return self.mlp(x)

    def compute_loss(self, logits, labels):
        """Cross-entropy loss for multiclass node classification."""
        return F.cross_entropy(logits, labels)
