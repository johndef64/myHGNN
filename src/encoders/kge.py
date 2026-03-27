"""
KGE Encoders: classical knowledge graph embedding methods as drop-in encoder replacements.

All encoders implement the same interface as GNN encoders:
    forward(x_dict, edge_index, **kwargs) -> (node_emb, rel_emb)

Graph-specific constructor args (in_channels_dict, conv_hidden_channels, conv_num_layers)
are accepted but ignored — KGE methods learn from scratch via embedding tables.
Embedding dimension is taken from mlp_out_emb_size.

Recommended pairings (encoder → decoder):
    DistMultKGEEncoder  → DistMultDecoder   (classic DistMult)
    TransEEncoder       → TransEDecoder     (TransE, Bordes 2013)
    RotatEEncoder       → RotatEDecoder     (RotatE, Sun 2019)
    Node2VecEncoder     → DistMultDecoder   (Node2Vec pre-train + DistMult scoring)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGEEncoder(nn.Module):
    """Base KGE encoder: learnable entity and relation embeddings, no message passing.

    Accepts the same constructor signature as GNN encoders for plug-in compatibility,
    but ignores graph-specific arguments.
    """

    def __init__(
        self,
        in_channels_dict,
        mlp_out_emb_size: int,
        conv_hidden_channels,
        num_nodes_per_type,
        num_entities: int,
        num_relations: int,
        conv_num_layers: int,
        device,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = mlp_out_emb_size
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device

        self.entity_emb = nn.Embedding(num_entities, mlp_out_emb_size)
        self.rel_emb = nn.Embedding(num_relations, mlp_out_emb_size)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.entity_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel_emb.weight, -0.1, 0.1)

    def forward(self, x_dict, edge_index, **kwargs):
        """Returns (entity_emb, rel_emb) without any message passing."""
        return self.drop(self.entity_emb.weight), self.rel_emb.weight


class DistMultKGEEncoder(KGEEncoder):
    """DistMult encoder: pure embedding lookup for h·r·t bilinear scoring.

    Identical to KGEEncoder base. Pair with DistMultDecoder (already in pipeline).
    This is the pure KGE version of DistMult, without GNN preprocessing.
    """
    pass


class TransEEncoder(KGEEncoder):
    """TransE encoder with standard init and entity L2-normalization during forward.

    Pair with TransEDecoder for score = gamma - ||h + r - t||_p.
    """

    def _reset_parameters(self):
        bound = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.entity_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        with torch.no_grad():
            # TransE paper: normalize relation embeddings to unit sphere at init
            self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data, p=2, dim=-1)

    def forward(self, x_dict, edge_index, **kwargs):
        # Entity embeddings are L2-normalized at each step (TransE paper §4)
        ent = F.normalize(self.entity_emb.weight, p=2, dim=-1)
        return self.drop(ent), self.rel_emb.weight


class RotatEEncoder(nn.Module):
    """RotatE encoder: entity embeddings in complex space, relations as rotation phases.

    Entity emb: [N, 2 * emb_dim]  — real and imaginary parts concatenated
    Relation emb: [R, emb_dim]    — phase angles, initialized in [-π, π]

    Pair with RotatEDecoder for score = gamma - ||h ∘ r - t||.
    """

    def __init__(
        self,
        in_channels_dict,
        mlp_out_emb_size: int,
        conv_hidden_channels,
        num_nodes_per_type,
        num_entities: int,
        num_relations: int,
        conv_num_layers: int,
        device,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = mlp_out_emb_size
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device
        self.gamma = kwargs.get('gamma', 12.0)   # needed for embedding_range init

        # Complex entity embeddings stored as [N, 2*d] (real | imag)
        self.entity_emb = nn.Embedding(num_entities, 2 * mlp_out_emb_size)
        # Relation rotation phases stored in [-pi, pi]
        self.rel_emb = nn.Embedding(num_relations, mlp_out_emb_size)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # RotatE paper (Sun et al. 2019): initialize entity embeddings in
        # [-embedding_range, embedding_range] so that the initial sum of
        # per-dimension complex moduli is comparable to gamma.
        # embedding_range = (gamma + epsilon) / d  → distances ≈ gamma at init.
        epsilon = 2.0
        embedding_range = (self.gamma + epsilon) / self.emb_dim
        nn.init.uniform_(self.entity_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.rel_emb.weight, -math.pi, math.pi)

    def forward(self, x_dict, edge_index, **kwargs):
        # No hard constraint on entity moduli: entities grow naturally during
        # training.  weight_decay must be 0 for RotatE parameters to prevent
        # entity embeddings collapsing toward zero (which makes all distances
        # collapse to 0 and all scores equal gamma → random ranking).
        return self.drop(self.entity_emb.weight), self.rel_emb.weight


class Node2VecEncoder(KGEEncoder):
    """Node2Vec encoder: entity embeddings pre-trained via biased random walks.

    Treats the KG as a homogeneous graph for random walks (relation types ignored).
    Relation embeddings are learnable and initialized randomly.
    Pre-training runs lazily on the first forward() call, then embeddings are
    fine-tuned end-to-end with the downstream decoder.

    Pair with DistMultDecoder.

    Requires: torch_geometric (pip install torch-geometric)
    If torch_geometric is unavailable, falls back to random initialization silently.
    """

    def __init__(
        self,
        in_channels_dict,
        mlp_out_emb_size: int,
        conv_hidden_channels,
        num_nodes_per_type,
        num_entities: int,
        num_relations: int,
        conv_num_layers: int,
        device,
        dropout: float = 0.0,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        pretrain_epochs: int = 5,
        pretrain_lr: float = 0.01,
        pretrain_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__(
            in_channels_dict, mlp_out_emb_size, conv_hidden_channels,
            num_nodes_per_type, num_entities, num_relations, conv_num_layers,
            device, dropout,
        )
        self._n2v_params = dict(
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
        )
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_batch_size = pretrain_batch_size
        self._pretrained = False

    def _pretrain(self, edge_index):
        """Run Node2Vec pre-training. Called once on first forward()."""
        try:
            from torch_geometric.nn import Node2Vec
        except ImportError:
            print("[!] Node2VecEncoder: torch_geometric not found — skipping pre-training, using random init.")
            self._pretrained = True
            return

        # KG edge_index is [E, 3] (src, rel, dst) → convert to [2, E] for PyG
        if edge_index.dim() == 2 and edge_index.shape[1] == 3:
            ei = edge_index[:, [0, 2]].t().contiguous().to(self.device)
        else:
            ei = edge_index.to(self.device)

        n2v = Node2Vec(
            ei,
            embedding_dim=self.emb_dim,
            **self._n2v_params,
            sparse=True,
        ).to(self.device)

        loader = n2v.loader(batch_size=self.pretrain_batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=self.pretrain_lr)

        n2v.train()
        for epoch in range(self.pretrain_epochs):
            total = 0.0
            for pos_rw, neg_rw in loader:
                opt.zero_grad()
                loss = n2v.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"[Node2Vec pre-train] epoch {epoch + 1}/{self.pretrain_epochs}  loss={total:.4f}")

        with torch.no_grad():
            self.entity_emb.weight.data.copy_(n2v.embedding.weight.data)
        self._pretrained = True

    def forward(self, x_dict, edge_index, **kwargs):
        if not self._pretrained:
            self._pretrain(edge_index)
        return self.drop(self.entity_emb.weight), self.rel_emb.weight
