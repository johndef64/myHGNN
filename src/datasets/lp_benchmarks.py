"""
Standard KG link prediction benchmark loaders.

Supports FB15k-237, WN18RR, and ogbl-biokg.
Returns the same dict format as load_kg_dataset() so the training loop
in train_link_prediction.py works without modification.
"""

import os
import numpy as np
import torch

from src.utils import graph_to_undirect, add_self_loops

LP_BENCHMARK_DATASETS = ['fb15k-237', 'wn18rr', 'ogbl-biokg']


def load_lp_benchmark(name, root='dataset/', device='cpu'):
    """
    Load a standard KG link prediction benchmark.

    Args:
        name: One of 'fb15k-237', 'wn18rr', 'ogbl-biokg'.
        root: Directory where datasets will be downloaded/cached.
        device: Target device ('cpu' or 'cuda').

    Returns:
        dict matching the format returned by load_kg_dataset().
    """
    name = name.lower()
    # Each benchmark gets its own subdirectory so their raw/ and processed/ folders
    # don't collide when multiple datasets are loaded with the same root.
    if name == 'fb15k-237':
        return _load_fb15k237(os.path.join(root, 'fb15k-237'), device)
    elif name == 'wn18rr':
        return _load_wn18rr(os.path.join(root, 'wn18rr'), device)
    elif name == 'ogbl-biokg':
        return _load_ogbl_biokg(os.path.join(root, 'ogbl-biokg'), device)
    else:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from: {LP_BENCHMARK_DATASETS}")


def _load_fb15k237(root, device):
    from torch_geometric.datasets import FB15k_237

    print('[i] Loading FB15k-237 (train/val/test splits)...')
    train_data = FB15k_237(root=root, split='train')[0]
    val_data   = FB15k_237(root=root, split='val')[0]
    test_data  = FB15k_237(root=root, split='test')[0]

    # edge_index: [2, N], edge_type: [N]
    def _to_triplets(data):
        ei = data.edge_index  # [2, N]
        et = data.edge_type   # [N]
        # stack as (head, relation, tail)
        return torch.stack([ei[0], et, ei[1]], dim=1).long()

    train_t = _to_triplets(train_data)
    val_t   = _to_triplets(val_data)
    test_t  = _to_triplets(test_data)

    num_original_relations = 237

    return _build_lp_output(train_t, val_t, test_t, num_original_relations)


def _load_wn18rr(root, device):
    """
    Load WN18RR directly from raw TSV files, bypassing PyG's WordNet18RR
    loader which has a known parsing bug with some versions of the downloaded file.
    Downloads the raw files from the canonical source if not present.
    """
    import urllib.request

    raw_dir = os.path.join(root, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    base_url = ('https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding'
                '/master/WN18RR/original/')
    splits = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}

    for split, fname in splits.items():
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            url = base_url + fname
            print(f'[i] Downloading WN18RR {split} from {url}')
            urllib.request.urlretrieve(url, fpath)

    print('[i] Loading WN18RR...')

    def _parse_file(fpath):
        triples = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    parts = line.split()   # fallback: whitespace-separated
                if len(parts) != 3:
                    continue
                triples.append(parts)
        return triples

    train_raw = _parse_file(os.path.join(raw_dir, 'train.txt'))
    val_raw   = _parse_file(os.path.join(raw_dir, 'valid.txt'))
    test_raw  = _parse_file(os.path.join(raw_dir, 'test.txt'))

    all_raw = train_raw + val_raw + test_raw

    # Build entity and relation id maps from the full corpus
    ent2id = {}
    rel2id = {}
    for h, r, t in all_raw:
        ent2id.setdefault(h, len(ent2id))
        ent2id.setdefault(t, len(ent2id))
        rel2id.setdefault(r, len(rel2id))

    def _to_tensor(raw):
        return torch.tensor(
            [[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in raw],
            dtype=torch.long
        )

    train_t = _to_tensor(train_raw)
    val_t   = _to_tensor(val_raw)
    test_t  = _to_tensor(test_raw)

    num_original_relations = len(rel2id)  # should be 11
    print(f'[i] WN18RR: {len(ent2id)} entities, {num_original_relations} relations')

    return _build_lp_output(train_t, val_t, test_t, num_original_relations)


def _load_ogbl_biokg(root, device):
    from ogb.linkproppred import PygLinkPropPredDataset

    print('[i] Loading ogbl-biokg (this may take a while on first run)...')
    print('[!] Warning: ogbl-biokg is large (~58K entities, 51 relations, 4.7M train triples). '
          'Filtered evaluation will be slow.')

    dataset = PygLinkPropPredDataset(name='ogbl-biokg', root=root)
    split_edge = dataset.get_edge_split()

    def _split_to_triplets(split_dict):
        heads     = torch.tensor(split_dict['head'],     dtype=torch.long)
        relations = torch.tensor(split_dict['relation'], dtype=torch.long)
        tails     = torch.tensor(split_dict['tail'],     dtype=torch.long)
        return torch.stack([heads, relations, tails], dim=1)

    train_t = _split_to_triplets(split_edge['train'])
    val_t   = _split_to_triplets(split_edge['valid'])
    test_t  = _split_to_triplets(split_edge['test'])

    num_original_relations = 51

    # Try to get num_nodes from the dataset; fall back to max IDs
    data = dataset[0]
    declared_num_nodes = getattr(data, 'num_nodes', 0)
    if not declared_num_nodes:
        print('[!] data.num_nodes unavailable, computing from max entity IDs.')

    return _build_lp_output(train_t, val_t, test_t, num_original_relations,
                            declared_num_nodes=declared_num_nodes)


def _build_lp_output(train_t, val_t, test_t, num_original_relations,
                     declared_num_nodes=0):
    """
    Shared helper that converts raw split tensors into the standard output dict.

    Args:
        train_t: LongTensor [N_train, 3] with columns (head, relation, tail).
        val_t:   LongTensor [N_val,   3]
        test_t:  LongTensor [N_test,  3]
        num_original_relations: Number of original (non-augmented) relation types R.
        declared_num_nodes: If > 0, use this as num_entities; otherwise compute
                            from max entity ID across all splits.

    Returns:
        dict matching load_kg_dataset() output format.
    """
    # num_relations follows the project convention: orig + reverse + self_loop
    num_relations = num_original_relations * 2 + 1

    # Compute num_entities safely from max entity ID across all splits
    all_triplets_cat = torch.cat([train_t, val_t, test_t], dim=0)
    max_from_data = int(all_triplets_cat[:, [0, 2]].max().item()) + 1
    if declared_num_nodes and declared_num_nodes >= max_from_data:
        num_entities = declared_num_nodes
    else:
        if declared_num_nodes and declared_num_nodes < max_from_data:
            print(f'[!] declared num_nodes={declared_num_nodes} < max entity ID+1={max_from_data}; '
                  f'using {max_from_data}.')
        num_entities = max_from_data

    # Build message-passing graph: undirected train edges + self-loops
    # graph_to_undirect expects a numpy array [N, 3] and returns a torch.Tensor
    train_np = train_t.numpy()
    train_index = graph_to_undirect(train_np, num_relations)
    train_index = add_self_loops(train_index, num_entities, num_relations)

    train_val_triplets      = torch.cat([train_t, val_t], dim=0)
    train_val_test_triplets = torch.cat([train_t, val_t, test_t], dim=0)

    return {
        'in_channels_dict':       {'node': None},
        'num_nodes_per_type':     {'node': num_entities},
        'num_entities':           num_entities,
        'num_relations':          num_relations,
        'train_triplets':         train_t,
        'train_index':            train_index,
        'flattened_features':     {'node': None},
        'val_triplets':           val_t,
        'train_val_triplets':     train_val_triplets,
        'test_triplets':          test_t,
        'train_val_test_triplets': train_val_test_triplets,
        'edge_index':             None,
        'ent2id':                 None,
        'relation2id':            None,
    }
