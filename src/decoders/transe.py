"""
TransE decoder for link prediction.

Scores (head, relation, tail) triples as:
    score(h, r, t) = gamma - ||h + r - t||_p

Positive triples get high scores (close to gamma); negatives get lower scores.
Pair with TransEEncoder for correct TransE semantics.
"""

import torch
import torch.nn as nn


class TransEDecoder(nn.Module):
    """
    TransE scoring function.

    Args:
        gamma:  Margin / score upper bound (paper default: 12.0 for FB15k-237).
        p_norm: Distance norm — 1 (L1, TransE paper) or 2 (L2).
    """

    def __init__(self, gamma: float = 12.0, p_norm: int = 1, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.p_norm = p_norm

    def forward(self, node_embeddings, rel_embeddings, triplets):
        """
        Args:
            node_embeddings: [num_entities, d]
            rel_embeddings:  [num_relations, d]
            triplets:        [B, 3] LongTensor (head, rel, tail)
        Returns:
            scores: [B] raw logits (higher = more likely)
        """
        h = node_embeddings[triplets[:, 0]]
        r = rel_embeddings[triplets[:, 1]]
        t = node_embeddings[triplets[:, 2]]
        return self.gamma - torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def reg_loss(self, node_embeddings, rel_embeddings, triplets, lambda_reg=0.01):
        """L2 regularization on batch embeddings."""
        h = node_embeddings[triplets[:, 0]]
        r = rel_embeddings[triplets[:, 1]]
        t = node_embeddings[triplets[:, 2]]
        return lambda_reg * (h.pow(2).mean() + r.pow(2).mean() + t.pow(2).mean())
