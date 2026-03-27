"""
Train and evaluate GNN models for GRAPH CLASSIFICATION.

Uses modular encoder/decoder architecture:
  - Encoder: CompGCN (adapted for batched graphs) --> sbagliato
  - Decoder: GraphClassifier (pooling + MLP)

Supports molecular benchmarks: MUTAG, PTC_MR, PROTEINS, NCI1 (from TUDataset).

# ENCODER DEFINITO DIRETTAMENTE NEL FILE
È un R-GCN custom con basis decomposition progettato per graph classification su batch di grafi eterogenei.

--> implementare l'uso di dataset custom in formato TSV (triple head-rel-tail) 

Usage:
  python train_graph_classification.py --dataset mutag --epochs 100
  python train_graph_classification.py --dataset ptc_mr  --epochs 200 --runs 10
  python train_graph_classification.py --dataset proteins --pooling mean_max --runs 5
  python train_graph_classification.py --dataset nci1 --pooling mean_max 
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from sklearn.metrics import accuracy_score, f1_score

from src.decoders.graph_classifier import GraphClassifier
from src.datasets.graph_cls_datasets import (
    load_graph_cls_dataset, collate_graphs, GRAPH_CLS_DATASETS
)
from src.utils import set_seed

warnings.simplefilter(action='ignore')

BASE_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Lightweight relational encoder for batched graphs ────────────────────────

class BatchRelationalEncoder(nn.Module):
    """
    A lightweight relational GNN encoder designed for batched graph classification.

    Unlike the KG encoders (CompGCN/RGCN), this encoder handles variable-size
    graphs in a batch. Uses basis decomposition similar to R-GCN but operates
    on batched node features directly.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_bases=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Relational convolution layers (basis decomposition)
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.conv_layers.append(
                RelationalConv(in_dim, out_dim, num_relations, num_bases, dropout)
            )
            self.norms.append(nn.LayerNorm(out_dim))

        self.dropout = dropout

    def forward(self, node_features, edge_triples, num_nodes, **kwargs):
        """
        Args:
            node_features: [total_nodes, in_channels]
            edge_triples: [total_edges, 3] (src, rel, dst)
            num_nodes: total number of nodes in the batch

        Returns:
            (node_embeddings, None)  — None for rel_emb (not used in graph cls)
        """
        x = self.input_proj(node_features)

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_triples, num_nodes)
            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x, None


class RelationalConv(nn.Module):
    """Simple relational convolution with basis decomposition."""
    def __init__(self, in_channels, out_channels, num_relations, num_bases, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        self.basis = nn.Parameter(torch.Tensor(self.num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.root = nn.Linear(in_channels, out_channels, bias=True)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.basis, gain=gain)
        nn.init.xavier_uniform_(self.att, gain=gain)

    def forward(self, x, edge_triples, num_nodes):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_triples: [num_edges, 3] (src, rel, dst)
            num_nodes: int
        """
        if edge_triples.size(0) == 0:
            return self.root(x)

        src = edge_triples[:, 0]
        rel = edge_triples[:, 1].long()
        dst = edge_triples[:, 2]

        # Weight decomposition: [num_relations, in, out]
        weights = torch.einsum('rb, bio -> rio', self.att, self.basis)

        # Compute messages per edge
        src_emb = x[src]  # [num_edges, in]
        rel_weights = weights[rel]  # [num_edges, in, out]
        messages = torch.bmm(src_emb.unsqueeze(1), rel_weights).squeeze(1)  # [num_edges, out]

        # Compute messages grouped by relation (avoids per-edge bmm)
        # messages = torch.zeros(src.size(0), self.out_channels, device=x.device)
        # for r in range(self.num_relations):
        #     mask = (rel == r)
        #     if mask.any():
        #         messages[mask] = x[src[mask]] @ weights[r]


        # Normalize by in-degree
        deg = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_inv = deg.pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0
        norm = deg_inv[dst].unsqueeze(1)
        messages = messages * norm

        # Aggregate
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # Self-loop
        out = out + self.root(x)

        return out


# ─── Training step ───────────────────────────────────────────────────────────

def train_step(encoder, decoder, optimizer, grad_norm,
               graphs, batch_size, dataset_info):
    encoder.train()
    decoder.train()

    indices = np.random.permutation(len(graphs))
    total_loss = 0
    all_preds, all_labels = [], []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]

        triples, features, batch_vec, labels, num_nodes, num_rels = collate_graphs(
            batch_graphs, device
        )

        optimizer.zero_grad()

        # Handle missing features
        if features is None:
            features = torch.zeros(num_nodes, 1, device=device)

        node_emb, _ = encoder(features, triples, num_nodes)
        logits = decoder(node_emb, None, batch_vec)
        loss = decoder.compute_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), grad_norm
        )
        optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return {'Loss': total_loss / len(graphs), 'Accuracy': acc}


# ─── Evaluation step ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(encoder, decoder, graphs, batch_size):
    encoder.eval()
    decoder.eval()

    all_preds, all_labels = [], []
    total_loss = 0

    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]

        triples, features, batch_vec, labels, num_nodes, num_rels = collate_graphs(
            batch_graphs, device
        )

        if features is None:
            features = torch.zeros(num_nodes, 1, device=device)

        node_emb, _ = encoder(features, triples, num_nodes)
        logits = decoder(node_emb, None, batch_vec)
        loss = decoder.compute_loss(logits, labels)

        total_loss += loss.item() * len(batch_graphs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        'Loss': total_loss / len(graphs),
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Macro-F1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'Micro-F1': f1_score(all_labels, all_preds, average='micro', zero_division=0),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    print(f'[i] Device: {device}')
    print(f'[i] Dataset: {args.dataset}')

    all_run_metrics = []

    for run_i in range(args.runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'\n{"="*60}\n[i] Run {run_i}/{args.runs} | seed={seed}')

        # Load dataset
        ds = load_graph_cls_dataset(args.dataset, root=args.data_root, seed=seed)
        graphs = ds['graphs']
        train_graphs = [graphs[i] for i in ds['train_indices']]
        val_graphs = [graphs[i] for i in ds['val_indices']]
        test_graphs = [graphs[i] for i in ds['test_indices']]

        in_channels = max(ds['num_node_features'], 1)  # at least 1
        num_edge_types = ds['num_edge_types']
        num_classes = ds['num_classes']

        print(f"[i] Graphs: {len(graphs)} (train={len(train_graphs)}, "
              f"val={len(val_graphs)}, test={len(test_graphs)})")
        print(f"[i] Node features: {in_channels}, Edge types: {num_edge_types}, "
              f"Classes: {num_classes}")

        # Build model
        encoder = BatchRelationalEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=args.hidden_dim,
            num_relations=num_edge_types,
            num_bases=min(args.num_bases, num_edge_types),
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        decoder = GraphClassifier(
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)

        # Optimizer
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Training loop
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        save_dir = None
        if not args.dry_run:
            ts = time.strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join('models', f'gc_{args.dataset}_{ts}')
            os.makedirs(save_dir, exist_ok=True)

        with trange(1, args.epochs + 1, desc=f'Run {run_i}') as pbar:
            for epoch in pbar:
                train_m = train_step(
                    encoder, decoder, optimizer, args.grad_norm,
                    train_graphs, args.batch_size, ds
                )
                scheduler.step()

                val_m = {'Accuracy': 0, 'Loss': 0}
                if epoch % args.evaluate_every == 0:
                    val_m = eval_step(encoder, decoder, val_graphs, args.batch_size)

                    if val_m['Accuracy'] > best_val_acc:
                        best_val_acc = val_m['Accuracy']
                        best_state = {
                            'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                            'decoder': {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                        }
                        patience_counter = 0
                    else:
                        patience_counter += args.evaluate_every
                        if args.early_stopping and patience_counter >= args.patience:
                            print(f"[i] Early stopping at epoch {epoch}")
                            break

                pbar.set_postfix(
                    loss=train_m['Loss'],
                    train_acc=train_m['Accuracy'],
                    val_acc=val_m.get('Accuracy', 0),
                )

        # Load best and test
        if best_state is not None:
            encoder.load_state_dict({k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict({k: v.to(device) for k, v in best_state['decoder'].items()})

        test_m = eval_step(encoder, decoder, test_graphs, args.batch_size)
        print(f"Run {run_i} | Test Accuracy: {test_m['Accuracy']:.4f}, "
              f"Macro-F1: {test_m['Macro-F1']:.4f}")

        all_run_metrics.append(test_m)

        if save_dir:
            torch.save(best_state, os.path.join(save_dir, f'best_run{run_i}.pt'))

    # Summary
    if all_run_metrics:
        keys = ['Accuracy', 'Macro-F1', 'Micro-F1']
        avg = {k: float(np.mean([m[k] for m in all_run_metrics])) for k in keys}
        std = {k: float(np.std([m[k] for m in all_run_metrics])) for k in keys}
        print(f"\n{'='*60}")
        print("SUMMARY:")
        for k in keys:
            print(f"  {k}: {avg[k]:.4f} ± {std[k]:.4f}")

        if save_dir:
            with open(os.path.join(save_dir, 'results.json'), 'w') as f:
                json.dump({'runs': all_run_metrics, 'avg': avg, 'std': std}, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Classification on Heterogeneous Graphs')
    parser.add_argument('--dataset', type=str, default='mutag',
                        help=f'Dataset: {GRAPH_CLS_DATASETS}')
    parser.add_argument('--data_root', type=str, default='dataset/')
    parser.add_argument('-r', '--runs', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-p', '--patience', type=int, default=30)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--evaluate_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_bases', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'sum', 'mean_max'])
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()
    main(args)


"""
Miligoramento


| Modifica          | Guadagno atteso        | Comandos                                                          |
| ----------------- | ---------------------- | ----------------------------------------------------------------- |
| --runs 10         | ±1-2% stabilità        | python train_graph_classification.py --dataset proteins --runs 10 |
| --numbases 2      | +0.5% (meno parametri) | --numbases 2                                                      |
| --pooling meanmax | +1-2%                  | --pooling meanmax                                                 |
| --hiddendim 128   | +1-3%                  | --hiddendim 128                                                   |
| Tutte insieme     | ~77-78%                | SOTA level!                                                       |
"""