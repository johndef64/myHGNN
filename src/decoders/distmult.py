"""
DistMult decoder for link prediction on knowledge graphs.

Scores (head, relation, tail) triples using element-wise product:
    score(h, r, t) = sum(h * r * t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMultDecoder(nn.Module):
    """
    DistMult scoring function for link prediction.

    Takes entity embeddings and relation embeddings from an encoder,
    and scores candidate triples.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, node_embeddings, rel_embeddings, triplets):
        """
        Score a batch of (head, relation, tail) triples.

        Args:
            node_embeddings: [num_entities, dim] entity embeddings from encoder.
            rel_embeddings: [num_relations, dim] relation embeddings from encoder.
            triplets: [batch_size, 3] LongTensor of (head_id, rel_id, tail_id).

        Returns:
            scores: [batch_size] raw logits.
        """
        s = node_embeddings[triplets[:, 0]]
        r = rel_embeddings[triplets[:, 1]]
        o = node_embeddings[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def score_loss(self, scores, target):
        """Binary cross-entropy loss with logits."""
        return F.binary_cross_entropy_with_logits(scores, target)

    def reg_loss(self, node_embeddings, rel_embeddings, triplets, lambda_reg=0.01):
        """L2 regularization on entity and relation embeddings involved in the batch."""
        s = node_embeddings[triplets[:, 0]]
        p = rel_embeddings[triplets[:, 1]]
        o = node_embeddings[triplets[:, 2]]
        return lambda_reg * (s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean())
