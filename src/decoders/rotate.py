"""
RotatE decoder for link prediction.

Scores (head, relation, tail) triples as:
    score(h, r, t) = gamma - ||h ∘ r - t||

where ∘ is the complex Hadamard product (element-wise rotation in complex space).

Entity embeddings must be [N, 2*d]:  first d dims = real part, last d dims = imaginary part.
Relation embeddings must be [R, d]:  rotation phases in radians (e.g. in [-π, π]).

Pair with RotatEEncoder for correct RotatE semantics.
Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation
           in Complex Space", ICLR 2019.
"""

import torch
import torch.nn as nn


class RotatEDecoder(nn.Module):
    """
    RotatE scoring function.

    Args:
        gamma: Margin / score upper bound (paper default: 12.0).
    """

    def __init__(self, gamma: float = 12.0, **kwargs):
        super().__init__()
        self.gamma = gamma

    def forward(self, node_embeddings, rel_embeddings, triplets):
        """
        Args:
            node_embeddings: [N, 2*d]  complex entity embeddings (real | imag)
            rel_embeddings:  [R, d]    relation rotation phases
            triplets:        [B, 3]    LongTensor (head, rel, tail)
        Returns:
            scores: [B] raw logits
        """
        d = rel_embeddings.shape[-1]

        h = node_embeddings[triplets[:, 0]]    # [B, 2d]
        r_phase = rel_embeddings[triplets[:, 1]]  # [B, d]
        t = node_embeddings[triplets[:, 2]]    # [B, 2d]

        h_re, h_im = h[..., :d], h[..., d:]
        t_re, t_im = t[..., :d], t[..., d:]

        # Relation as unit complex: e^{i * phase}
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # Complex multiplication: h ∘ r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # ||h ∘ r - t|| summed over d complex dimensions
        diff = torch.stack([hr_re - t_re, hr_im - t_im], dim=-1)  # [B, d, 2]
        score = self.gamma - diff.norm(p=2, dim=-1).sum(dim=-1)    # [B]
        return score

    def reg_loss(self, node_embeddings, rel_embeddings, triplets, lambda_reg=0.01):
        """L2 regularization on batch embeddings."""
        h = node_embeddings[triplets[:, 0]]
        r = rel_embeddings[triplets[:, 1]]
        t = node_embeddings[triplets[:, 2]]
        return lambda_reg * (h.pow(2).mean() + r.pow(2).mean() + t.pow(2).mean())
