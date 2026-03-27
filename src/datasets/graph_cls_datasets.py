"""
Graph classification datasets.

Supports molecular graph benchmarks (MUTAG, PTC_MR, PROTEINS, NCI1) from TUDataset,
which have typed edges (bond types) making them heterogeneous.

Each graph is converted to our triple format [src, rel, dst] for encoder compatibility.

Dorvebbe supportare anche dataset custom in formato TSV (triple head-rel-tail) 
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

GRAPH_CLS_DATASETS = ['mutag', 'ptc_mr', 'proteins', 'nci1']


def load_graph_cls_dataset(name, root='dataset/', seed=42, val_ratio=0.1, test_ratio=0.1):
    """
    Load a graph classification dataset.

    Args:
        name: Dataset name ('mutag', 'ptc_mr', 'proteins', 'nci1').
        root: Root directory for downloading/caching.
        seed: Random seed for splits.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.

    Returns:
        dict with keys:
            graphs: list of GraphData objects
            num_node_features: int
            num_edge_types: int
            num_classes: int
            train_indices, val_indices, test_indices: lists of graph indices
    """
    name_upper = name.upper() if name.lower() != 'ptc_mr' else 'PTC_MR'
    return _load_tudataset(name_upper, root, seed, val_ratio, test_ratio)


class GraphData:
    """
    Container for a single graph in our framework's format.

    Attributes:
        edge_triples: [num_edges, 3] tensor (src, rel, dst)
        node_features: [num_nodes, feat_dim] tensor or None
        num_nodes: int
        num_relations: int
        label: int (graph class)
    """
    def __init__(self, edge_triples, node_features, num_nodes, num_relations, label):
        self.edge_triples = edge_triples
        self.node_features = node_features
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.label = label


def _load_tudataset(name, root, seed, val_ratio, test_ratio):
    """Load from PyG's TUDataset and convert to our format."""
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=os.path.join(root, 'tu_datasets'), name=name, use_node_attr=True)

    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features

    # Determine number of edge types
    max_edge_type = 0
    has_edge_attr = False
    for data in dataset:
        if data.edge_attr is not None:
            has_edge_attr = True
            if data.edge_attr.dim() == 1:
                max_edge_type = max(max_edge_type, int(data.edge_attr.max().item()))
            else:
                # One-hot encoded edge attributes
                max_edge_type = max(max_edge_type, data.edge_attr.shape[1] - 1)
    num_edge_types = max_edge_type + 1 if has_edge_attr else 1

    # Convert each PyG graph to our format
    graphs = []
    for data in dataset:
        num_nodes = data.num_nodes
        src = data.edge_index[0]
        dst = data.edge_index[1]

        if has_edge_attr and data.edge_attr is not None:
            if data.edge_attr.dim() == 1:
                rel = data.edge_attr.long()
            else:
                rel = data.edge_attr.argmax(dim=1).long()
        else:
            rel = torch.zeros(src.size(0), dtype=torch.long)

        edge_triples = torch.stack([src, rel, dst], dim=1)

        node_features = data.x if data.x is not None else None

        graphs.append(GraphData(
            edge_triples=edge_triples,
            node_features=node_features,
            num_nodes=num_nodes,
            num_relations=num_edge_types,
            label=int(data.y.item())
        ))

    # Train/val/test split
    rng = np.random.RandomState(seed)
    n = len(graphs)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:n_train + n_val].tolist()
    test_indices = perm[n_train + n_val:].tolist()

    return {
        'graphs': graphs,
        'num_node_features': num_node_features,
        'num_edge_types': num_edge_types,
        'num_classes': num_classes,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
    }


def collate_graphs(graph_list, device='cpu'):
    """
    Batch multiple GraphData objects into a single batched graph.

    Offsets node indices so all graphs live in one large graph.
    Returns a batch assignment vector for pooling.

    Args:
        graph_list: list of GraphData objects.
        device: target device.

    Returns:
        batched_triples: [total_edges, 3] (src, rel, dst) with offset node IDs
        batched_features: [total_nodes, feat_dim] or None
        batch_vector: [total_nodes] graph assignment for each node
        labels: [num_graphs] class labels
        num_nodes_total: int
        num_relations: int (max across batch)
    """
    all_triples = []
    all_features = []
    all_batch = []
    all_labels = []
    offset = 0
    max_relations = 0

    for graph_idx, g in enumerate(graph_list):
        triples = g.edge_triples.clone()
        triples[:, 0] += offset  # offset src
        triples[:, 2] += offset  # offset dst
        all_triples.append(triples)

        if g.node_features is not None:
            all_features.append(g.node_features)

        all_batch.append(torch.full((g.num_nodes,), graph_idx, dtype=torch.long))
        all_labels.append(g.label)
        max_relations = max(max_relations, g.num_relations)
        offset += g.num_nodes

    batched_triples = torch.cat(all_triples, dim=0).to(device)
    batch_vector = torch.cat(all_batch, dim=0).to(device)
    labels = torch.tensor(all_labels, dtype=torch.long).to(device)

    if all_features:
        batched_features = torch.cat(all_features, dim=0).to(device)
    else:
        batched_features = None

    return batched_triples, batched_features, batch_vector, labels, offset, max_relations
