"""
Benchmark script: Heterogeneous GNN methods (R-GCN, R-GAT, CompGCN) for link prediction
on standard KG benchmarks: FB15k-237 and WN18RR.

Uses BCE + label smoothing 0.1 (as in the CompGCN paper), paper-standard hyperparameters
(config lp-benchmark = 200-dim embeddings).
Saves results to reports/gnn_lp_benchmark_<timestamp>.{json,txt}

Paper reference values (filtered, full-graph ranking):

  FB15k-237 (from Vashishth et al. 2020, CompGCN — ICLR):
    R-GCN+DistMult : MRR=0.249  H@1=0.151  H@3=0.264  H@10=0.417
    CompGCN+DistMult: MRR=0.355  H@1=0.264  H@3=0.390  H@10=0.535

  WN18RR (from Vashishth et al. 2020, CompGCN — ICLR):
    CompGCN+DistMult: MRR=0.479  H@1=0.443  H@3=0.494  H@10=0.546

Notes:
  - R-GCN is evaluated with DistMult decoder (as in CompGCN paper setup).
  - R-GAT does not have widely reported results on these benchmarks; results here
    serve as a novel comparison point.
  - R-GCN original paper (Schlichtkrull et al. 2018) used FB15k and WN18
    (non-standard splits), so direct comparison is not straightforward.
  - All models use filtered full-graph ranking (standard for these benchmarks).
  - ogbl-biokg uses type-constrained evaluation (too large for full-graph ranking).

Usage:
  python benchmark_gnn_lp.py                              # all models, fb15k-237 + wn18rr
  python benchmark_gnn_lp.py --runs 3 --epochs 500
  python benchmark_gnn_lp.py --models compgcn --benchmarks wn18rr
  python benchmark_gnn_lp.py --models rgcn rgat --benchmarks fb15k-237
  python benchmark_gnn_lp.py --config lp-benchmark-64    # faster, 64-dim
  python benchmark_gnn_lp.py --benchmarks ogbl-biokg --models compgcn
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import torch
from tqdm.auto import trange

# Reuse building blocks from the existing training script
from train_link_prediction import (
    build_model,
    train_step,
    eval_step,
    device,
)
from src.datasets.lp_benchmarks import load_lp_benchmark
from src.utils import set_seed, negative_sampling_filtered

warnings.simplefilter('ignore')

BASE_SEED = 42

# ─── Paper reference values ───────────────────────────────────────────────────
# Sources:
#   R-GCN:    Schlichtkrull et al. 2018 (ESWC) — original paper uses FB15k/WN18,
#             not FB15k-237/WN18RR. Values below are from the CompGCN paper
#             (Vashishth et al. 2020, ICLR), Table 3 — R-GCN re-evaluated with
#             DistMult decoder on FB15k-237.
#   R-GAT:    No standard widely cited values on these benchmarks.
#   CompGCN:  Vashishth et al. 2020 (ICLR), Table 3 — CompGCN + DistMult decoder.
PAPER_RESULTS = {
    'fb15k-237': {
        'rgcn':    {'MRR': 0.249, 'Hits@1': 0.151, 'Hits@3': 0.264, 'Hits@10': 0.417},
        'compgcn': {'MRR': 0.355, 'Hits@1': 0.264, 'Hits@3': 0.390, 'Hits@10': 0.535},
        # R-GAT: no canonical reference value — novel comparison
    },
    'wn18rr': {
        'compgcn': {'MRR': 0.479, 'Hits@1': 0.443, 'Hits@3': 0.494, 'Hits@10': 0.546},
        # R-GCN and R-GAT: not widely reported on WN18RR with DistMult decoder
    },
    'ogbl-biokg': {
        # ogbl-biokg uses type-constrained evaluation; CompGCN paper does not report
        # on this benchmark. OGB leaderboard values vary by implementation.
    },
}

ALL_BENCHMARKS = ['fb15k-237', 'wn18rr']
ALL_MODELS = ['rgcn', 'rgat', 'compgcn']


# ─── Encoder kwargs for R-GAT ─────────────────────────────────────────────────

def _make_encoder_kwargs(model_name, train_index):
    """
    R-GAT requires edges sorted by relation type and a change_points vector.
    All other models: no extra kwargs.

    Args:
        model_name: str
        train_index: LongTensor [N, 3] — (head, relation, tail) message-passing edges.

    Returns:
        (sorted_train_index, encoder_kwargs dict)
    """
    if model_name != 'rgat':
        return train_index, {}

    # Sort by relation id so R-GAT can process each relation block contiguously
    sort_idx = train_index[:, 1].argsort()
    train_index = train_index[sort_idx]
    rel_ids = train_index[:, 1]
    change_points = torch.cat([
        torch.tensor([0], device=train_index.device),
        (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
        torch.tensor([rel_ids.size(0)], device=train_index.device),
    ])
    return train_index, {'change_points': change_points}


# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(benchmark, model_name, config_name, runs, epochs, patience,
                   neg_batch_size, evaluate_every, negative_rate):
    """Train and evaluate one (benchmark, model) combination. Returns metrics dict."""

    loss_kwargs = dict(
        loss_fn='bce',
        label_smoothing=0.1,            # as in CompGCN paper
        alpha=0.25, gamma=3.0, alpha_adv=2.0,   # focal params, ignored with loss='bce'
    )

    # FB15k-237 and WN18RR are homogeneous (one entity type) → full-graph filtered ranking
    # ogbl-biokg is heterogeneous and large → type-constrained ranking
    use_type_constrained = (benchmark == 'ogbl-biokg')

    all_run_metrics = []

    for run_i in range(runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'  [Run {run_i+1}/{runs}] seed={seed}')

        ds = load_lp_benchmark(benchmark, root='dataset/', device=device)

        encoder, decoder, mp = build_model(
            model_name, ds, config_name, './src/models_params.json'
        )
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        train_index_raw = torch.tensor(ds['train_index']).to(device)
        train_index, encoder_kwargs = _make_encoder_kwargs(model_name, train_index_raw)

        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}
        train_triplets          = ds['train_triplets']
        val_triplets            = ds['val_triplets']
        test_triplets           = ds['test_triplets']
        train_val_triplets      = ds['train_val_triplets'].to(device)
        train_val_test_triplets = ds['train_val_test_triplets'].to(device)

        all_entities_arr = np.arange(ds['num_entities'])
        all_true_arr     = train_val_test_triplets.cpu().numpy()
        reg_param        = mp.get('regularization', 1e-5)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            all_params, lr=mp['learning_rate'], weight_decay=mp['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=mp.get('scheduler_gamma', 0.995)
        )

        best_val_mrr = -1.0
        last_improvement = 0
        best_state = None
        rng_batch = np.random.RandomState(seed)

        n_train = len(train_triplets)
        use_minibatch = neg_batch_size > 0 and n_train > neg_batch_size
        steps_per_epoch = (n_train + neg_batch_size - 1) // neg_batch_size if use_minibatch else 1

        with trange(1, epochs + 1, desc=f'    {benchmark}/{model_name}') as pbar:
            for epoch in pbar:
                if use_minibatch:
                    """
                    per ogni epoch:
                        ├── se neg_batch_size > 0:
                        │     shuffle del dataset
                        │     per ogni mini-batch (tutti):  ← nuovo inner loop
                        │         neg sampling + train_step
                        └── altrimenti:
                            1 step su tutto il dataset
                        
                        scheduler.step()  ← 1 volta per epoch
                        ogni evaluate_every epoch: validation

                    Avvertimento: con neg_batch_size piccolo su WN18RR, ogni epoch fa ~42 gradient step (87K/2048), ognuno con una forward pass sull'intero grafo (il message passing gira sempre sull'intera struttura). Questo è 42x più lento per epoch rispetto al full batch. Il guadagno in memoria c'è, ma il tempo aumenta.

                    Per WN18RR usa comunque --neg_batch_size 0 (full batch) — è veloce e non va in OOM. Il mini-batch con inner loop ha senso su FB15k-237 se 272K × 51 = 14M triple dovesse andare in OOM.
                    """
                    # Shuffle once per epoch, then iterate over all mini-batches
                    perm = rng_batch.permutation(n_train)
                    train_m = None
                    for step in range(steps_per_epoch):
                        start = step * neg_batch_size
                        end = min(start + neg_batch_size, n_train)
                        batch_triplets = train_triplets[perm[start:end]]

                        neg_trips, neg_labels = negative_sampling_filtered(
                            batch_triplets, all_entities_arr, negative_rate,
                            all_true_arr, seed=seed + epoch * steps_per_epoch + step
                        )
                        neg_trips, neg_labels = neg_trips.to(device), neg_labels.to(device)

                        train_m = train_step(
                            encoder, decoder, optimizer, mp['grad_norm'], reg_param,
                            features, train_index, neg_trips, neg_labels,
                            **loss_kwargs, **encoder_kwargs,
                        )
                else:
                    # Full-batch: 1 step covers all training triples
                    neg_trips, neg_labels = negative_sampling_filtered(
                        train_triplets, all_entities_arr, negative_rate,
                        all_true_arr, seed=seed + epoch
                    )
                    neg_trips, neg_labels = neg_trips.to(device), neg_labels.to(device)

                    train_m = train_step(
                        encoder, decoder, optimizer, mp['grad_norm'], reg_param,
                        features, train_index, neg_trips, neg_labels,
                        **loss_kwargs, **encoder_kwargs,
                    )

                scheduler.step()  # once per epoch regardless of mini-batch count

                # Validation: sampled MRR (fast proxy during training)
                if epoch % evaluate_every == 0:
                    val_neg, val_lab = negative_sampling_filtered(
                        val_triplets, all_entities_arr, negative_rate, all_true_arr, seed=seed + 1000
                    )
                    val_neg, val_lab = val_neg.to(device), val_lab.to(device)

                    val_m = eval_step(
                        encoder, decoder, reg_param,
                        features, train_index, val_neg, val_lab, train_val_triplets,
                        **loss_kwargs,
                        eval_filtered=False,   # sampled during training for speed
                        use_type_constrained=use_type_constrained,
                        **encoder_kwargs,
                    )
                    if val_m.get('MRR', 0) > best_val_mrr:
                        best_val_mrr = val_m['MRR']
                        last_improvement = epoch
                        best_state = {
                            'encoder': {k: v.cpu().clone()
                                        for k, v in encoder.state_dict().items()},
                            'decoder': {k: v.cpu().clone()
                                        for k, v in decoder.state_dict().items()},
                        }
                    elif (epoch - last_improvement) >= patience:
                        print(f'\n    [i] Early stopping at epoch {epoch}')
                        break

                    pbar.set_postfix(
                        loss=f"{train_m['Loss']:.4f}",
                        val_loss=f"{val_m['Loss']:.4f}",
                        val_mrr=f"{val_m.get('MRR', 0):.4f}",
                    )

        # Load best checkpoint
        if best_state is not None:
            encoder.load_state_dict({k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict({k: v.to(device) for k, v in best_state['decoder'].items()})

        # Test: full filtered ranking (all entities for FB15k-237/WN18RR,
        #       type-constrained for ogbl-biokg)
        test_neg, test_lab = negative_sampling_filtered(
            test_triplets, all_entities_arr, 1, all_true_arr, seed=seed + 2000
        )
        test_neg, test_lab = test_neg.to(device), test_lab.to(device)

        test_m = eval_step(
            encoder, decoder, reg_param,
            features, train_index, test_neg, test_lab, train_val_test_triplets,
            **loss_kwargs,
            eval_filtered=True,
            all_target_triplets=train_val_test_triplets,
            num_entities=ds['num_entities'],
            use_type_constrained=use_type_constrained,
            **encoder_kwargs,
        )

        run_result = {
            'MRR':    test_m['MRR'],
            'Hits@1': test_m['Hits@'][1],
            'Hits@3': test_m['Hits@'][3],
            'Hits@10':test_m['Hits@'][10],
            'Auroc':  test_m['Auroc'],
            'Auprc':  test_m['Auprc'],
        }
        print(f'    → MRR={run_result["MRR"]:.4f}  H@1={run_result["Hits@1"]:.4f}  '
              f'H@3={run_result["Hits@3"]:.4f}  H@10={run_result["Hits@10"]:.4f}')
        all_run_metrics.append(run_result)

    keys = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Auroc', 'Auprc']
    avg = {k: float(np.mean([m[k] for m in all_run_metrics])) for k in keys}
    std = {k: float(np.std ([m[k] for m in all_run_metrics])) for k in keys}
    return {'runs': all_run_metrics, 'avg': avg, 'std': std}


# ─── Report helpers ───────────────────────────────────────────────────────────

def _fmt_row(label, avg, std, paper=None):
    line = (f'  {label:<16} MRR={avg["MRR"]:.4f}±{std["MRR"]:.4f}  '
            f'H@1={avg["Hits@1"]:.4f}  H@3={avg["Hits@3"]:.4f}  H@10={avg["Hits@10"]:.4f}')
    if paper:
        line += (f'\n  {"(paper ref)":<16} MRR={paper["MRR"]:.4f}          '
                 f'H@1={paper["Hits@1"]:.4f}  H@3={paper["Hits@3"]:.4f}  '
                 f'H@10={paper["Hits@10"]:.4f}')
    return line


def build_text_report(results, args):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '=' * 70,
        'GNN Link Prediction Benchmark Results',
        f'Generated : {ts}',
        f'Config    : {args.config}',
        f'Epochs    : {args.epochs}  |  Patience: {args.patience}  |  Runs: {args.runs}',
        f'Device    : {device}',
        f'Loss      : BCE + label smoothing 0.1 (as in CompGCN paper)',
        '=' * 70,
        '',
        'Paper references:',
        '  R-GCN (FB15k-237)  — Vashishth et al. 2020, CompGCN (ICLR), Table 3',
        '  CompGCN (FB15k-237, WN18RR) — Vashishth et al. 2020, CompGCN (ICLR), Table 3',
        '  R-GAT — no canonical reference on these benchmarks (novel baseline)',
        '=' * 70,
    ]
    for benchmark in args.benchmarks:
        lines.append(f'\n{benchmark.upper()}')
        lines.append('-' * len(benchmark))
        if benchmark == 'ogbl-biokg':
            lines.append('  [type-constrained evaluation — full-graph ranking not feasible]')
        for model in args.models:
            key = f'{benchmark}/{model}'
            if key not in results:
                lines.append(f'  {model:<16} [skipped / failed]')
                continue
            r = results[key]
            if 'error' in r:
                lines.append(f'  {model:<16} [ERROR: {r["error"]}]')
                continue
            paper = PAPER_RESULTS.get(benchmark, {}).get(model)
            lines.append(_fmt_row(model, r['avg'], r['std'], paper))
    lines.append('\n' + '=' * 70)
    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs('reports', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    json_path = f'reports/gnn_lp_benchmark_{ts}.json'
    txt_path  = f'reports/gnn_lp_benchmark_{ts}.txt'

    print(f'[i] Device    : {device}')
    print(f'[i] Benchmarks: {args.benchmarks}')
    print(f'[i] Models    : {args.models}')
    print(f'[i] Config    : {args.config}')
    print(f'[i] Epochs    : {args.epochs}  Patience: {args.patience}  Runs: {args.runs}')
    print(f'[i] Loss      : BCE + label smoothing 0.1')
    print()

    results = {}
    total = len(args.benchmarks) * len(args.models)
    done = 0

    for benchmark in args.benchmarks:
        for model in args.models:
            done += 1
            print(f'\n[{done}/{total}] {benchmark} / {model}')
            print('-' * 50)
            try:
                r = run_experiment(
                    benchmark=benchmark,
                    model_name=model,
                    config_name=args.config,
                    runs=args.runs,
                    epochs=args.epochs,
                    patience=args.patience,
                    neg_batch_size=args.neg_batch_size,
                    evaluate_every=args.evaluate_every,
                    negative_rate=args.negative_rate,
                )
                results[f'{benchmark}/{model}'] = r

                # Save intermediate results after each experiment
                with open(json_path, 'w') as f:
                    json.dump({'args': vars(args), 'results': results}, f, indent=2)

            except Exception as e:
                print(f'  [ERROR] {e}')
                results[f'{benchmark}/{model}'] = {'error': str(e)}

    # Final report
    report_txt = build_text_report(results, args)
    print('\n' + report_txt)

    with open(json_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)

    print(f'\n[i] Report saved to:\n    {json_path}\n    {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Link Prediction Benchmark')
    parser.add_argument('--benchmarks', nargs='+', default=ALL_BENCHMARKS,
                        choices=['fb15k-237', 'wn18rr', 'ogbl-biokg'],
                        help='Benchmarks to test (default: fb15k-237 wn18rr)')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='GNN models to test (default: rgcn rgat compgcn)')
    parser.add_argument('--config', type=str, default='lp-benchmark',
                        help='Config name from models_params.json '
                             '(default: lp-benchmark = 200-dim, as in CompGCN paper)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs per experiment (default: 1)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max training epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stopping patience in epochs (default: 200)')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Validate every N epochs (default: 10)')
    parser.add_argument('--neg_batch_size', type=int, default=0,
                        help='Positive batch size before neg sampling per epoch. '
                             '0=full batch (recommended for WN18RR/FB15k-237). '
                             'Use e.g. 4096 only if you get OOM. (default: 0)')
    parser.add_argument('--negative_rate', type=int, default=50,
                        help='Negatives per positive during training. '
                             'Higher = better training signal but more memory. '
                             'CompGCN paper uses 1-vs-All (~14K for FB15k-237). (default: 50)')
    args = parser.parse_args()
    main(args)
