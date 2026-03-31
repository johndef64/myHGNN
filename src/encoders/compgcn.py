"""
CompGCN Encoder for Heterogeneous Knowledge Graphs.

Extracted from hetero_compgcn.py — encoder only, no decoder methods.
Returns (node_embeddings, relation_embeddings) from forward().
"""

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch.nn.modules.module import Module


class CompGCNConv(Module):
    """
    Compositional Graph Convolutional Layer.
    Performs message passing using compositional operators on entity
    and relation embeddings (Vashishth et al., 2020).
    """
    def __init__(self, in_channels, out_channels, num_relations,
                 comp_fn='mult', dropout=0.0, bias=True, edge_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.comp_fn = comp_fn
        self.dropout = dropout
        self.edge_norm = edge_norm

        self.w_loop = nn.Linear(in_channels, out_channels, bias=False)
        self.w_in = nn.Linear(in_channels, out_channels, bias=False)
        self.w_out = nn.Linear(in_channels, out_channels, bias=False)
        self.w_rel = nn.Linear(in_channels, out_channels, bias=False)

        self.loop_rel = nn.Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.w_loop.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_in.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_out.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_rel.weight, gain=gain)
        nn.init.xavier_uniform_(self.loop_rel, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compositional_op(self, ent, rel):
        if self.comp_fn == 'mult':
            return ent * rel
        elif self.comp_fn == 'sub':
            return ent - rel
        elif self.comp_fn == 'corr':
            return ent * rel
        else:
            raise ValueError(f"Unsupported composition: {self.comp_fn}")

    def forward(self, node_emb, rel_emb, edge_index):
        s = edge_index[:, 0]
        r = edge_index[:, 1].long()
        t = edge_index[:, 2]

        num_nodes = node_emb.size(0)

        if self.edge_norm:
            t_deg = torch_scatter.scatter_add(
                torch.ones_like(t, dtype=torch.float, device=t.device),
                t, dim=0, dim_size=num_nodes
            )
            t_deg_inv = t_deg.pow(-0.5)
            t_deg_inv[torch.isinf(t_deg_inv)] = 0

            s_deg = torch_scatter.scatter_add(
                torch.ones_like(s, dtype=torch.float, device=s.device),
                s, dim=0, dim_size=num_nodes
            )
            s_deg_inv = s_deg.pow(-0.5)
            s_deg_inv[torch.isinf(s_deg_inv)] = 0
        else:
            t_deg_inv = s_deg_inv = None

        # Forward edges
        comp_out = self.compositional_op(node_emb[s], rel_emb[r])
        msg_out = self.w_out(comp_out)
        if self.edge_norm:
            msg_out = msg_out * s_deg_inv[s].unsqueeze(1)
        agg_out = torch_scatter.scatter_add(msg_out, t, dim=0, dim_size=num_nodes)
        if self.edge_norm:
            agg_out = agg_out * t_deg_inv.unsqueeze(1)

        # Inverse edges
        comp_in = self.compositional_op(node_emb[t], rel_emb[r])
        msg_in = self.w_in(comp_in)
        if self.edge_norm:
            msg_in = msg_in * t_deg_inv[t].unsqueeze(1)
        agg_in = torch_scatter.scatter_add(msg_in, s, dim=0, dim_size=num_nodes)
        if self.edge_norm:
            agg_in = agg_in * s_deg_inv.unsqueeze(1)

        # Self-loops
        loop_rel_expand = self.loop_rel.expand(num_nodes, -1)
        comp_loop = self.compositional_op(node_emb, loop_rel_expand)
        msg_loop = self.w_loop(comp_loop)

        out = (agg_out + agg_in + msg_loop) / 3.0
        if self.bias is not None:
            out = out + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Transform and return relation embeddings for the next layer
        updated_rel_emb = self.w_rel(rel_emb)

        return out, updated_rel_emb


class CompGCNEncoder(nn.Module):
    """
    Heterogeneous CompGCN encoder for knowledge graphs with multiple node types.

    Returns:
        (node_embeddings, relation_embeddings) tuple from forward().
    """
    def __init__(self, in_channels_dict, mlp_out_emb_size, conv_hidden_channels,
                 num_nodes_per_type, num_entities, num_relations, conv_num_layers,
                 opn='sub', dropout=0.1, activation_function=F.relu,
                 use_layer_norm=True, edge_norm=True, device='cuda:0'):
        super().__init__()

        self.mlp_out = mlp_out_emb_size
        self.device = device
        self.num_entities = num_entities
        self.num_nodes_per_type = num_nodes_per_type
        self.activation_function = activation_function
        self.layers_num = conv_num_layers
        self.use_layer_norm = use_layer_norm

        # Node type embedding projection
        self.mlp_dict = nn.ModuleDict()
        self.node_type_offset = {}
        offset = 0
        for node_type, in_channels in in_channels_dict.items():
            self.node_type_offset[node_type] = offset
            if in_channels is None:
                self.mlp_dict[node_type] = nn.Embedding(num_nodes_per_type[node_type], mlp_out_emb_size)
            else:
                self.mlp_dict[node_type] = nn.Sequential(
                    nn.Linear(in_channels, mlp_out_emb_size),
                    nn.Dropout(dropout)
                )
            offset += num_nodes_per_type[node_type]

        # Single initial relation embeddings (including inverses) — refined through layers like the original CompGCN
        self.init_rel = nn.Parameter(torch.Tensor(2 * num_relations, mlp_out_emb_size))
        nn.init.xavier_uniform_(self.init_rel)

        # CompGCN layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        for idx in range(conv_num_layers):
            in_channels = conv_hidden_channels[f'layer_{idx-1}'] if idx > 0 else mlp_out_emb_size
            out_channels = conv_hidden_channels[f'layer_{idx}']
            self.conv_layers.append(
                CompGCNConv(in_channels, out_channels, num_relations,
                            comp_fn=opn, dropout=dropout, edge_norm=edge_norm)
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_channels))

    def forward(self, x_dict, edge_index, **kwargs):
        """
        Returns:
            (node_embeddings [num_entities, hidden_dim],
             relation_embeddings [num_relations, hidden_dim])
        """
        x_all = torch.zeros(self.num_entities, self.mlp_out, device=self.device)
        for node_type, features in x_dict.items():
            offset = self.node_type_offset[node_type]
            if features is None:
                emb = self.mlp_dict[node_type](
                    torch.arange(0, self.num_nodes_per_type[node_type], device=self.device)
                )
            else:
                emb = self.mlp_dict[node_type](features)
            x_all[offset:offset + self.num_nodes_per_type[node_type]] = emb

        x = x_all
        rel_emb = self.init_rel  # relations flow and are refined through layers
        for layer_idx in range(self.layers_num):
            x, rel_emb = self.conv_layers[layer_idx](x, rel_emb, edge_index)
            if layer_idx < self.layers_num - 1:
                x = self.activation_function(x)
            if self.use_layer_norm:
                x = self.layer_norms[layer_idx](x)

        return x, rel_emb
