"""
R-GAT Encoder for Heterogeneous Knowledge Graphs.

Extracted from hetero_rgat.py — encoder only, no decoder methods.
Returns (node_embeddings, relation_embeddings) from forward().
"""

import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class HRGATConv(Module):
    """
    Heterogeneous Relational Graph Attention Convolution layer.
    Uses basis decomposition with per-relation attention.
    """
    def __init__(self, in_channels, out_channels, every_node, every_relation,
                 num_bases, no_bias=False, no_attention=False, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.every_node = every_node
        self.every_relation = every_relation
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(every_relation, num_bases))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)

        if no_attention:
            self.register_parameter('attention', None)
        else:
            self.attention = nn.Parameter(torch.Tensor(every_relation, 2 * out_channels))
        if no_bias:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.initialise_weights()

    def initialise_weights(self):
        weight_gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.basis, gain=weight_gain)
        nn.init.xavier_uniform_(self.att, gain=weight_gain)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.attention, gain=weight_gain)

    def forward(self, node_embeddings, triples, change_points, device):
        weights = torch.einsum('rb, bio -> rio', self.att, self.basis)
        Wh = torch.einsum('rio, ni -> rno', weights, node_embeddings).contiguous()
        h_prime = torch.zeros(self.every_node, self.out_channels, device=device).contiguous()

        edge_type = torch.arange(0, self.every_relation, device=device, dtype=torch.int)

        for rel in edge_type:
            if rel + 1 >= len(change_points):
                continue
            start = change_points[rel]
            end = change_points[rel + 1]
            if start >= end:
                continue
            rel_index = triples[start:end][:, [0, 2]]
            if rel_index.size(0) == 0:
                continue

            a = self.attention[rel]
            Wh_rel = Wh[rel]

            # Attention coefficients (standard GAT: LeakyReLU(a^T [Wh_i || Wh_j]))
            Wh_concat = torch.cat((Wh_rel[rel_index[:, 0], :], Wh_rel[rel_index[:, 1], :]), dim=1).t()
            e_raw = self.leakyrelu(a[None, :].mm(Wh_concat).squeeze())

            # Per-destination-node softmax for numerical stability
            dst = rel_index[:, 1]
            e_max = torch.zeros(self.every_node, device=device)
            e_max.scatter_reduce_(0, dst, e_raw, reduce='amax', include_self=False)
            e_raw = e_raw - e_max[dst]
            edge_e = torch.exp(e_raw)

            # Normalize per destination node (softmax over incoming edges)
            attn_sum = torch.zeros(self.every_node, device=device)
            attn_sum.scatter_add_(0, dst, edge_e)
            normalized_attention = edge_e / torch.clamp(attn_sum[dst], min=1e-8)
            normalized_attention = self.attn_drop(normalized_attention)

            # Aggregate: weighted sum of transformed neighbor features
            h_prime_rel = torch_sparse.spmm(
                rel_index.T, normalized_attention,
                self.every_node, self.every_node, Wh_rel
            )
            h_prime = h_prime + h_prime_rel

        output = h_prime + self.bias
        if node_embeddings.size(-1) == self.out_channels:
            output = output + node_embeddings
        return output


class RGATEncoder(nn.Module):
    """
    Heterogeneous R-GAT encoder for knowledge graphs with multiple node types.

    Note: forward() requires `change_points` kwarg for relation-boundary indices.

    Returns:
        (node_embeddings, relation_embeddings) tuple from forward().
    """
    def __init__(self, in_channels_dict, mlp_hidden_channels_dict, mlp_out_emb_size,
                 conv_hidden_channels, num_nodes_per_type, num_entities, num_relations,
                 conv_num_layers, num_bases, activation_function=F.relu, dropout=0.1,
                 device='cuda:0'):
        super().__init__()

        self.mlp_out = mlp_out_emb_size
        self.device = device
        self.num_nodes_per_type = num_nodes_per_type
        self.activation_function = activation_function
        self.layers_num = conv_num_layers

        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_relations, conv_hidden_channels[f"layer_{conv_num_layers-1}"])
        )
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.mlp_dict = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            if in_channels is None:
                self.mlp_dict[node_type] = nn.Embedding(num_nodes_per_type[node_type], mlp_out_emb_size)
            else:
                self.mlp_dict[node_type] = nn.Linear(in_channels, mlp_out_emb_size)

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            HRGATConv(mlp_out_emb_size, conv_hidden_channels['layer_0'],
                      num_entities, num_relations, num_bases, dropout=dropout)
        )
        for idx in range(1, conv_num_layers):
            self.conv_layers.append(
                HRGATConv(conv_hidden_channels[f'layer_{idx-1}'], conv_hidden_channels[f'layer_{idx}'],
                          num_entities, num_relations, num_bases, dropout=dropout)
            )

    def forward(self, x_dict, edge_index, change_points=None, **kwargs):
        """
        Args:
            x_dict: Dict mapping node types to feature tensors (or None for embeddings).
            edge_index: [num_edges, 3] tensor of (src, rel, dst) triples.
            change_points: Tensor of relation boundary indices (required for attention).

        Returns:
            (node_embeddings [num_entities, hidden_dim],
             relation_embeddings [num_relations, hidden_dim])
        """
        if change_points is None:
            raise ValueError("RGATEncoder requires 'change_points' argument")

        x_list = []
        for node_type, features in x_dict.items():
            if features is None:
                x_list.append(self.mlp_dict[node_type](
                    torch.arange(0, self.num_nodes_per_type[node_type],
                                 dtype=torch.long, device=self.device)
                ))
            else:
                x_list.append(self.mlp_dict[node_type](features))
        x = torch.cat(x_list)

        for layer in range(self.layers_num - 1):
            x = self.activation_function(
                self.conv_layers[layer](x, edge_index, change_points, self.device)
            )
        x = self.conv_layers[self.layers_num - 1](x, edge_index, change_points, self.device)

        return x, self.relation_embedding


# --- Sparse helpers ---

def stack_matrices(triples, nodes_num, num_rels):
    assert triples.dtype == torch.long
    R, n = num_rels, nodes_num
    size = (R * n, n)
    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    fr = offset + fr
    indices = torch.cat([fr[:, None], to[:, None]], dim=1)
    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[:, 0].max()}, {size}, {R}'
    assert indices[:, 1].max() < size[1], f'{indices[:, 1].max()}, {size}, {R}'
    return indices, size


def sum_sparse(indices, values, size, device):
    assert len(indices.size()) == len(values.size()) + 1
    k, _ = indices.size()
    ones = torch.ones((size[1], 1), device=device)
    values = torch.sparse_coo_tensor(indices=indices.t(), values=values, size=size,
                                     dtype=torch.float, device=device)
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]
    return sums.view(k)
