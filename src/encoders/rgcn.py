"""
R-GCN Encoder for Heterogeneous Knowledge Graphs.

Extracted from hetero_rgcn.py — encoder only, no decoder methods.
Returns (node_embeddings, relation_embeddings) from forward().
"""

import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class RGCNConv(Module):
    """
    Relational Graph Convolutional layer with basis decomposition.

    From "Modeling Relational Data with Graph Convolutional Networks"
    (Schlichtkrull et al., 2018).
    """
    def __init__(self, in_channels, out_channels, every_node, every_relation, num_bases, no_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.every_node = every_node
        self.every_relation = every_relation
        self.num_bases = num_bases
        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(every_relation, num_bases))
        self.leakyrelu = nn.LeakyReLU()
        if no_bias:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.initialise_weights()

    def initialise_weights(self):
        weight_gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.basis, gain=weight_gain)
        nn.init.xavier_uniform_(self.att, gain=weight_gain)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias)

    def forward(self, node_embeddings, triples, device):
        weights = torch.einsum('rb, bio -> rio', self.att, self.basis)
        adj_indices, adj_size = stack_matrices(triples, self.every_node, self.every_relation)
        vals = torch.ones(adj_indices.size(0), dtype=torch.float, device=device)
        sums = sum_sparse(adj_indices, vals, adj_size, device=device)
        vals = vals / sums
        af = torch_sparse.spmm(adj_indices.T, vals, adj_size[0], adj_size[1], node_embeddings)
        af = af.view(self.every_relation, adj_size[1], self.in_channels)
        output = torch.einsum('rio, rni -> no', weights, af)
        if self.bias is not None:
            output = torch.add(output, self.bias)
        return output


class RGCNEncoder(nn.Module):
    """
    Heterogeneous R-GCN encoder for knowledge graphs with multiple node types.

    Returns:
        (node_embeddings, relation_embeddings) tuple from forward().
    """
    def __init__(self, in_channels_dict, mlp_hidden_channels_dict, mlp_out_emb_size,
                 conv_hidden_channels, num_nodes_per_type, num_entities, num_relations,
                 conv_num_layers, num_bases, activation_function=F.relu, device='cuda:0'):
        super().__init__()

        self.mlp_out = mlp_out_emb_size
        self.device = device
        self.num_nodes_per_type = num_nodes_per_type
        self.activation_function = activation_function
        self.layers_num = conv_num_layers

        # Relation embeddings for decoder
        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_relations, conv_hidden_channels[f"layer_{conv_num_layers-1}"])
        )
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # Per-type input projection
        self.mlp_dict = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            if in_channels is None:
                self.mlp_dict[node_type] = nn.Embedding(num_nodes_per_type[node_type], mlp_out_emb_size)
            else:
                self.mlp_dict[node_type] = nn.Linear(in_channels, mlp_out_emb_size)

        # R-GCN layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            RGCNConv(mlp_out_emb_size, conv_hidden_channels['layer_0'],
                     num_entities, num_relations, num_bases)
        )
        for idx in range(1, conv_num_layers):
            self.conv_layers.append(
                RGCNConv(conv_hidden_channels[f'layer_{idx-1}'], conv_hidden_channels[f'layer_{idx}'],
                         num_entities, num_relations, num_bases)
            )

    def forward(self, x_dict, edge_index, **kwargs):
        """
        Returns:
            (node_embeddings [num_entities, hidden_dim],
             relation_embeddings [num_relations, hidden_dim])
        """
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
            x = self.activation_function(self.conv_layers[layer](x, edge_index, self.device))
        x = self.conv_layers[self.layers_num - 1](x, edge_index, self.device)

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
