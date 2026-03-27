"""
Node classification datasets for heterogeneous graphs.

Supports RDF benchmark datasets (AIFB, MUTAG, BGS, AM) from PyG,
and custom TSV-based KGs with external node labels.

All datasets are converted to a unified format compatible with our encoders:
  - edge triples: [num_edges, 3] tensor of (src, rel_type, dst)
  - node labels + train/test masks
"""

import os
import torch
import numpy as np
import pandas as pd

NODE_CLS_DATASETS = ['aifb', 'mutag', 'bgs', 'am']


def load_node_cls_dataset(name, root='dataset/', seed=42, val_ratio=0.1):
    """
    Load a node classification dataset.

    Args:
        name: Dataset name ('aifb', 'mutag', 'bgs', 'am') or path to TSV.
        root: Root directory for downloading/caching datasets.
        seed: Random seed for train/val split.
        val_ratio: Fraction of training nodes for validation.

    Returns:
        dict with keys:
            edge_triples: [num_edges, 3] LongTensor (src, rel, dst)
            num_entities: int
            num_relations: int
            num_classes: int
            in_channels_dict: {node_type: feature_dim or None}
            num_nodes_per_type: {node_type: count}
            flattened_features: {node_type: tensor or None}
            train_idx, val_idx, test_idx: LongTensors of node indices
            train_y, val_y, test_y: LongTensors of labels
    """
    name_lower = name.lower()
    if name_lower in NODE_CLS_DATASETS:
        return _load_pyg_entities(name_lower, root, seed, val_ratio)
    elif os.path.exists(name):
        return _load_tsv_node_cls(name, root, seed, val_ratio)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {NODE_CLS_DATASETS} or provide a TSV path."
        )


def _load_pyg_entities(name, root, seed, val_ratio):
    """
    Load AIFB/MUTAG/BGS/AM from PyTorch Geometric's Entities datasets.
    Converts PyG format to our triple-based format.
    """
    from torch_geometric.datasets import Entities

    dataset = Entities(root=os.path.join(root, 'entities'), name=name.upper())
    data = dataset[0]

    # PyG Entities format:
    #   data.edge_index: [2, num_edges]
    #   data.edge_type: [num_edges]
    #   data.train_idx, data.train_y
    #   data.test_idx, data.test_y
    #   data.num_nodes

    num_entities = data.num_nodes
    num_relations = int(data.edge_type.max().item()) + 1

    # Convert to [num_edges, 3] triple format: (src, rel, dst)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    rel = data.edge_type
    edge_triples = torch.stack([src, rel, dst], dim=1).long()

    # Train/test split from dataset; carve out validation from train
    train_idx = data.train_idx
    train_y = data.train_y
    test_idx = data.test_idx
    test_y = data.test_y

    # Split train into train + val
    rng = np.random.RandomState(seed)
    n_train = len(train_idx)
    n_val = max(1, int(n_train * val_ratio))
    perm = rng.permutation(n_train)
    val_mask = perm[:n_val]
    train_mask = perm[n_val:]

    val_idx_split = train_idx[val_mask]
    val_y_split = train_y[val_mask]
    train_idx_split = train_idx[train_mask]
    train_y_split = train_y[train_mask]

    num_classes = int(train_y.max().item()) + 1

    # All nodes are same type (no heterogeneous features in Entities datasets)
    in_channels_dict = {'node': None}
    num_nodes_per_type = {'node': num_entities}
    flattened_features = {'node': None}

    return {
        'edge_triples': edge_triples,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'num_classes': num_classes,
        'in_channels_dict': in_channels_dict,
        'num_nodes_per_type': num_nodes_per_type,
        'flattened_features': flattened_features,
        'train_idx': train_idx_split.long(),
        'val_idx': val_idx_split.long(),
        'test_idx': test_idx.long(),
        'train_y': train_y_split.long(),
        'val_y': val_y_split.long(),
        'test_y': test_y.long(),
    }


def _load_tsv_node_cls(tsv_path, root, seed, val_ratio):
    """
    Load a custom KG from TSV with external node labels.

    Expects:
        - tsv_path: path to triples TSV (head, interaction, tail)
        - A companion file <tsv_path>.labels.tsv with (node_id, label) columns
    """
    from src.utils import (
        load_data, entities2id_offset, rel2id_offset, edge_ind_to_id,
        entities_features_flattening, graph_to_undirect, add_self_loops
    )

    labels_path = tsv_path.replace('.tsv', '.labels.tsv')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Label file not found: {labels_path}. "
            f"For custom TSV node classification, provide a .labels.tsv file."
        )

    # Load graph
    edge_index_df, node_features_per_type = load_data(tsv_path, {}, quiet=True)
    ent2id, all_nodes_per_type = entities2id_offset(edge_index_df, node_features_per_type, quiet=True)
    relation2id = rel2id_offset(edge_index_df)
    indexed = edge_ind_to_id(edge_index_df, ent2id, relation2id)
    flattened_features = entities_features_flattening(node_features_per_type, all_nodes_per_type)

    num_nodes_per_type = {nt: len(nodes) for nt, nodes in all_nodes_per_type.items()}
    num_entities = sum(num_nodes_per_type.values())
    num_relations = len(relation2id)

    # Build edge triples
    triples = indexed[["head", "interaction", "tail"]].values
    triples = graph_to_undirect(triples, num_relations)
    triples = add_self_loops(triples, num_entities, num_relations)
    edge_triples = torch.tensor(triples).long()

    # Load labels
    labels_df = pd.read_csv(labels_path, sep='\t', header=0, names=['node_id', 'label'])
    label2id = {l: i for i, l in enumerate(sorted(labels_df['label'].unique()))}
    num_classes = len(label2id)

    node_ids = []
    node_labels = []
    for _, row in labels_df.iterrows():
        if row['node_id'] in ent2id:
            node_ids.append(ent2id[row['node_id']])
            node_labels.append(label2id[row['label']])

    node_ids = torch.tensor(node_ids, dtype=torch.long)
    node_labels = torch.tensor(node_labels, dtype=torch.long)

    # Random train/val/test split (60/10/30)
    rng = np.random.RandomState(seed)
    n = len(node_ids)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * 0.3))
    n_train = n - n_val - n_test

    train_idx = node_ids[perm[:n_train]]
    train_y = node_labels[perm[:n_train]]
    val_idx = node_ids[perm[n_train:n_train + n_val]]
    val_y = node_labels[perm[n_train:n_train + n_val]]
    test_idx = node_ids[perm[n_train + n_val:]]
    test_y = node_labels[perm[n_train + n_val:]]

    in_channels_dict = {
        nt: (feat.shape[1] if feat is not None else None)
        for nt, feat in flattened_features.items()
    }

    return {
        'edge_triples': edge_triples,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'num_classes': num_classes,
        'in_channels_dict': in_channels_dict,
        'num_nodes_per_type': num_nodes_per_type,
        'flattened_features': flattened_features,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_y': train_y,
        'val_y': val_y,
        'test_y': test_y,
    }
