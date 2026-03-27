"""
Benchmark script: GNN methods (R-GCN, R-GAT, CompGCN) AND classical KGE embedding
methods (DistMult, TransE, RotatE, Node2Vec) on standard node classification
benchmarks: AIFB, MUTAG, BGS, AM.

GNN methods do message passing; KGE methods learn pure entity embeddings (no message
passing) — useful to quantify how much graph structure/propagation actually helps.
DistMult/TransE/RotatE are trained end-to-end with cross-entropy (not pre-trained on
link prediction). Node2Vec pre-trains embeddings via biased random walks, then
fine-tunes the classification head.

Uses paper-standard hyperparameters (config node-classification: hidden=16, 2 layers,
weight_decay=5e-4 — as in R-GCN paper, Schlichtkrull et al. 2018).
Saves results to reports/node_cls_benchmark_<timestamp>.{json,txt}

Paper reference values (accuracy) — GNN models only:
  AIFB:  R-GCN = 95.83 %
  MUTAG: R-GCN = 73.23 %
  BGS:   R-GCN = 83.10 %
  AM:    R-GCN = 89.29 %
(No published baselines for KGE methods on these node classification benchmarks.)

Usage:
  python benchmark_node_cls.py                                   # tutti i modelli, 10 runs
  python benchmark_node_cls.py --runs 5 --epochs 200
  python benchmark_node_cls.py --models rgcn rgat compgcn        # solo GNN
  python benchmark_node_cls.py --models distmult_kge transe rotate node2vec  # solo KGE
  python benchmark_node_cls.py --models rgcn distmult_kge --datasets aifb mutag
  python benchmark_node_cls.py --config node-classification-32   # 32-dim variant
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange
from sklearn.metrics import accuracy_score, f1_score

# Reuse building blocks from the existing training script
from train_node_classification import build_model, train_step, eval_step, device
from src.datasets.node_cls_datasets import load_node_cls_dataset
from src.utils import set_seed, graph_to_undirect, add_self_loops

warnings.simplefilter('ignore')

BASE_SEED = 42

# ─── Paper reference values ───────────────────────────────────────────────────
# Source: Schlichtkrull et al. 2018 "Modeling Relational Data with Graph
# Convolutional Networks"  (Table 2, R-GCN entity classification, filtered).
# Weighted accuracy reported in the paper.
PAPER_RESULTS = {
    'aifb':  {'rgcn': {'Accuracy': 0.9583}},
    'mutag': {'rgcn': {'Accuracy': 0.7323}},
    'bgs':   {'rgcn': {'Accuracy': 0.8310}},
    'am':    {'rgcn': {'Accuracy': 0.8929}},
}

ALL_DATASETS   = ['aifb', 'mutag', 'bgs', 'am']
ALL_GNN_MODELS = ['rgcn', 'rgat', 'compgcn']
ALL_KGE_MODELS = ['distmult_kge', 'transe', 'rotate', 'node2vec']
ALL_MODELS     = ALL_GNN_MODELS + ALL_KGE_MODELS

# Models that are NOT GNNs (no message passing — ignore edge_index, no change_points)
_KGE_MODELS = set(ALL_KGE_MODELS)


# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(dataset_name, model_name, config_name, runs, epochs, patience,
                   evaluate_every, data_root):
    """Train and evaluate one (dataset, model) combination. Returns metrics dict."""

    all_run_metrics = []

    for run_i in range(runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'  [Run {run_i+1}/{runs}] seed={seed}')

        ds = load_node_cls_dataset(dataset_name, root=data_root, seed=seed)

        print(f'    Entities: {ds["num_entities"]}, Relations: {ds["num_relations"]}, '
              f'Classes: {ds["num_classes"]}')
        print(f'    Train: {len(ds["train_idx"])}, Val: {len(ds["val_idx"])}, '
              f'Test: {len(ds["test_idx"])}')

        # Build edge index (PyG Entities datasets already include both directions;
        # RGCN/RGAT need explicit reverse edges; CompGCN handles it internally)
        triples_np = ds['edge_triples'].numpy()
        if model_name in ('rgcn', 'rgat'):
            triples_np = graph_to_undirect(triples_np, ds['num_relations'])
            triples_np = add_self_loops(triples_np, ds['num_entities'], ds['num_relations'])
        edge_index = torch.tensor(triples_np, dtype=torch.long).to(device)

        # Update num_relations in ds to reflect actual edge types
        ds['num_relations'] = int(edge_index[:, 1].max().item()) + 1

        encoder, decoder, mp, _out_dim = build_model(
            model_name, ds, config_name, './src/models_params.json'
        )
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}

        train_idx = ds['train_idx'].to(device)
        train_y   = ds['train_y'].to(device)
        val_idx   = ds['val_idx'].to(device)
        val_y     = ds['val_y'].to(device)
        test_idx  = ds['test_idx'].to(device)
        test_y    = ds['test_y'].to(device)

        # RGAT requires edges sorted by relation type + change_points vector
        encoder_kwargs = {}
        if model_name == 'rgat':
            sort_idx = edge_index[:, 1].argsort()
            edge_index = edge_index[sort_idx]
            rel_ids = edge_index[:, 1]
            encoder_kwargs['change_points'] = torch.cat([
                torch.tensor([0], device=device),
                (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
                torch.tensor([rel_ids.size(0)], device=device),
            ])

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(
            all_params,
            lr=mp['learning_rate'],
            weight_decay=mp.get('weight_decay', 0.0),
        )

        best_val_acc = 0.0
        best_state   = None
        patience_counter = 0

        with trange(1, epochs + 1, desc=f'    {dataset_name}/{model_name}') as pbar:
            for epoch in pbar:
                train_m = train_step(
                    encoder, decoder, optimizer, mp.get('grad_norm', 1.0),
                    features, edge_index, train_idx, train_y,
                    **encoder_kwargs,
                )

                val_m = {'Accuracy': 0.0, 'Loss': 0.0}
                if epoch % evaluate_every == 0:
                    val_m = eval_step(
                        encoder, decoder, features, edge_index,
                        val_idx, val_y, **encoder_kwargs,
                    )

                    if val_m['Accuracy'] > best_val_acc:
                        best_val_acc = val_m['Accuracy']
                        patience_counter = 0
                        best_state = {
                            'encoder': {k: v.cpu().clone()
                                        for k, v in encoder.state_dict().items()},
                            'decoder': {k: v.cpu().clone()
                                        for k, v in decoder.state_dict().items()},
                        }
                    else:
                        patience_counter += evaluate_every
                        if patience_counter >= patience:
                            print(f'\n    [i] Early stopping at epoch {epoch}')
                            break

                pbar.set_postfix(
                    loss=f"{train_m['Loss']:.4f}",
                    train_acc=f"{train_m['Accuracy']:.4f}",
                    val_acc=f"{val_m['Accuracy']:.4f}",
                )

        # Load best model and evaluate on test set
        if best_state is not None:
            encoder.load_state_dict({k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict({k: v.to(device) for k, v in best_state['decoder'].items()})

        test_m = eval_step(
            encoder, decoder, features, edge_index,
            test_idx, test_y, **encoder_kwargs,
        )

        run_result = {
            'Accuracy': test_m['Accuracy'],
            'Macro-F1': test_m['Macro-F1'],
            'Micro-F1': test_m['Micro-F1'],
        }
        print(f'    → Acc={run_result["Accuracy"]:.4f}  '
              f'Macro-F1={run_result["Macro-F1"]:.4f}  '
              f'Micro-F1={run_result["Micro-F1"]:.4f}')
        all_run_metrics.append(run_result)

    keys = ['Accuracy', 'Macro-F1', 'Micro-F1']
    avg = {k: float(np.mean([m[k] for m in all_run_metrics])) for k in keys}
    std = {k: float(np.std ([m[k] for m in all_run_metrics])) for k in keys}
    return {'runs': all_run_metrics, 'avg': avg, 'std': std}


# ─── Report helpers ───────────────────────────────────────────────────────────

def _fmt_nc_row(label, avg, std, paper=None):
    line = (f'  {label:<16} Acc={avg["Accuracy"]:.4f}±{std["Accuracy"]:.4f}  '
            f'MacroF1={avg["Macro-F1"]:.4f}  MicroF1={avg["Micro-F1"]:.4f}')
    if paper and 'Accuracy' in paper:
        line += f'\n  {"(paper ref)":<16} Acc={paper["Accuracy"]:.4f}'
    return line


def build_text_report(results, args):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '=' * 70,
        'Node Classification Benchmark Results',
        f'Generated : {ts}',
        f'Config    : {args.config}',
        f'Epochs    : {args.epochs}  |  Patience: {args.patience}  |  Runs: {args.runs}',
        f'Device    : {device}',
        '=' * 70,
    ]
    for dataset in args.datasets:
        lines.append(f'\n{dataset.upper()}')
        lines.append('-' * len(dataset))
        for model in args.models:
            key = f'{dataset}/{model}'
            if key not in results:
                lines.append(f'  {model:<16} [skipped / failed]')
                continue
            r = results[key]
            if 'error' in r:
                lines.append(f'  {model:<16} [ERROR: {r["error"]}]')
                continue
            paper = PAPER_RESULTS.get(dataset, {}).get(model)
            lines.append(_fmt_nc_row(model, r['avg'], r['std'], paper))
    lines.append('\n' + '=' * 70)
    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs('reports', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    json_path = f'reports/node_cls_benchmark_{ts}.json'
    txt_path  = f'reports/node_cls_benchmark_{ts}.txt'

    print(f'[i] Device  : {device}')
    print(f'[i] Datasets: {args.datasets}')
    print(f'[i] Models  : {args.models}')
    print(f'[i] Config  : {args.config}')
    print(f'[i] Epochs  : {args.epochs}  Patience: {args.patience}  Runs: {args.runs}')
    print()

    results = {}
    total = len(args.datasets) * len(args.models)
    done = 0

    for dataset in args.datasets:
        for model in args.models:
            done += 1
            print(f'\n[{done}/{total}] {dataset} / {model}')
            print('-' * 50)
            try:
                r = run_experiment(
                    dataset_name=dataset,
                    model_name=model,
                    config_name=args.config,
                    runs=args.runs,
                    epochs=args.epochs,
                    patience=args.patience,
                    evaluate_every=args.evaluate_every,
                    data_root=args.data_root,
                )
                results[f'{dataset}/{model}'] = r

                # Save intermediate results after each experiment
                with open(json_path, 'w') as f:
                    json.dump({'args': vars(args), 'results': results}, f, indent=2)

            except Exception as e:
                print(f'  [ERROR] {e}')
                results[f'{dataset}/{model}'] = {'error': str(e)}

    # Final report
    report_txt = build_text_report(results, args)
    print('\n' + report_txt)

    with open(json_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)

    print(f'\n[i] Report saved to:\n    {json_path}\n    {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node Classification Benchmark')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        choices=ALL_DATASETS,
                        help='Datasets to test (default: aifb mutag bgs am)')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='Models to test. GNN: rgcn rgat compgcn. '
                             'KGE (no message passing): distmult_kge transe rotate node2vec. '
                             'Default: all models.')
    parser.add_argument('--config', type=str, default='node-classification',
                        help='Config name from models_params.json '
                             '(default: node-classification = hidden=16, paper-standard)')
    parser.add_argument('--data_root', type=str, default='dataset/',
                        help='Root directory for dataset download (default: dataset/)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per experiment (default: 10, as in R-GCN paper)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience in epochs (default: 30)')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Validate every N epochs (default: 1)')
    args = parser.parse_args()
    main(args)
