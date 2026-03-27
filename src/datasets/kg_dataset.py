"""
Knowledge Graph dataset loader for link prediction.

Loads TSV triples and converts them to the format expected by the encoders.
Wraps existing utility functions from src/utils.py.
"""

import os
import numpy as np
import torch

from src.utils import (
    load_data, entities2id_offset, rel2id_offset, edge_ind_to_id,
    entities_features_flattening, set_target_label, select_target_triplets,
    triple_sampling, graph_to_undirect, add_self_loops
)


def load_kg_dataset(tsv_path, task, validation_size=0.1, test_size=0.2,
                    oversample_rate=1, undersample_rate=1.0, seed=42,
                    quiet=False, device='cpu'):
    """
    Load a KG dataset from TSV for link prediction.

    Args:
        tsv_path: Path to TSV file with (head, interaction, tail) columns.
        task: Relation type(s) to predict (comma-separated string or list).
        validation_size: Fraction for validation split.
        test_size: Fraction for test split.
        oversample_rate: Repeat target triples N times in training.
        undersample_rate: Fraction of non-target triples to keep.
        seed: Random seed.
        quiet: Suppress output.
        device: Target device.

    Returns:
        dict with keys:
            in_channels_dict, num_nodes_per_type, num_entities, num_relations,
            train_triplets, train_index, flattened_features, val_triplets,
            train_val_triplets, test_triplets, train_val_test_triplets,
            edge_index, ent2id, relation2id
    """
    resolved = os.path.normpath(tsv_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    edge_index, node_features_per_type = load_data(resolved, {}, quiet)

    # Label target edges
    target_list = [x for x in task.split(',')] if isinstance(task, str) else task
    edge_index = set_target_label(edge_index, target_list)

    ent2id, all_nodes_per_type = entities2id_offset(edge_index, node_features_per_type, quiet)
    relation2id = rel2id_offset(edge_index)
    indexed_edge_index = edge_ind_to_id(edge_index, ent2id, relation2id)
    flattened_features = entities_features_flattening(node_features_per_type, all_nodes_per_type)

    indexed_edge_index["label"] = edge_index["label"].values
    non_target_triplets, target_triplets = select_target_triplets(indexed_edge_index)

    train_triplets, val_triplets, test_triplets = triple_sampling(
        target_triplets.loc[:, ["head", "interaction", "tail"]].values,
        validation_size, test_size, quiet, seed
    )

    # Under-sampling non-target
    if undersample_rate < 1.0:
        rnd = np.random.RandomState(seed)
        keep = int(len(non_target_triplets) * undersample_rate)
        idx = rnd.choice(len(non_target_triplets), size=keep, replace=False)
        non_target_triplets = non_target_triplets.iloc[idx]

    # Over-sampling training positives
    if oversample_rate > 1:
        train_triplets = np.repeat(train_triplets, oversample_rate, axis=0)

    non_target_triplets = non_target_triplets.loc[:, ["head", "interaction", "tail"]].values
    train_index = np.concatenate([non_target_triplets, train_triplets], axis=0)

    num_nodes_per_type = {nt: len(nodes) for nt, nodes in all_nodes_per_type.items()}
    num_entities = sum(num_nodes_per_type.values())
    num_relations = len(relation2id)

    train_index = graph_to_undirect(train_index, num_relations)
    train_index = add_self_loops(train_index, num_entities, num_relations)

    train_triplets = torch.tensor(train_triplets)
    val_triplets = torch.tensor(val_triplets)
    test_triplets = torch.tensor(test_triplets)
    train_val_triplets = torch.cat([train_triplets, val_triplets], 0)
    train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], 0)

    in_channels_dict = {
        nt: (feat.shape[1] if feat is not None else None)
        for nt, feat in flattened_features.items()
    }

    return {
        'in_channels_dict': in_channels_dict,
        'num_nodes_per_type': num_nodes_per_type,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'train_triplets': train_triplets,
        'train_index': train_index,
        'flattened_features': flattened_features,
        'val_triplets': val_triplets,
        'train_val_triplets': train_val_triplets,
        'test_triplets': test_triplets,
        'train_val_test_triplets': train_val_test_triplets,
        'edge_index': edge_index,
        'ent2id': ent2id,
        'relation2id': relation2id,
    }
