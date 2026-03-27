"""
Benchmark script: KGE classical methods (DistMult, TransE, RotatE) on FB15k-237 and WN18RR.

Uses paper-standard hyperparameters (config lp-benchmark, 200-dim, BCE + label smoothing).
Saves results to reports/kge_lp_benchmark_<timestamp>.{json,txt}

Paper reference values (filtered, full-graph ranking):
  FB15k-237: DistMult=0.241, TransE=0.294, RotatE=0.338  (MRR)
  WN18RR:    DistMult=0.430, TransE=0.226, RotatE=0.476  (MRR)

Note on loss function:
  - DistMult and RotatE use BCE + label smoothing 0.1 (as in CompGCN paper setup).
  - TransE uses BCE without smoothing. The original paper uses margin loss (not BCE),
    so TransE results may differ from paper values.
  - RotatE originally uses self-adversarial negative sampling; here we use filtered
    uniform sampling, which may yield slightly lower results than the original paper.

Usage:
  python benchmark_kge_lp.py                            # all models, fb15k-237 + wn18rr
  python benchmark_kge_lp.py --runs 3 --epochs 500
  python benchmark_kge_lp.py --models transe rotate --benchmarks wn18rr
  python benchmark_kge_lp.py --config lp-benchmark-64  # faster, 64-dim
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
#   DistMult: Yang et al. 2015 / Kadlec et al. 2017 (re-eval)
#   TransE:   Bordes et al. 2013 / re-evaluated with filtered ranking
#   RotatE:   Sun et al. 2019 (ICLR)
PAPER_RESULTS = {
    'fb15k-237': {
        'distmult_kge': {'MRR': 0.241, 'Hits@1': 0.155, 'Hits@3': 0.263, 'Hits@10': 0.419},
        'transe':        {'MRR': 0.294, 'Hits@1': 0.202, 'Hits@3': 0.326, 'Hits@10': 0.465},
        'rotate':        {'MRR': 0.338, 'Hits@1': 0.241, 'Hits@3': 0.375, 'Hits@10': 0.533},
    },
    'wn18rr': {
        'distmult_kge': {'MRR': 0.430, 'Hits@1': 0.390, 'Hits@3': 0.440, 'Hits@10': 0.490},
        'transe':        {'MRR': 0.226, 'Hits@1': 0.011, 'Hits@3': 0.401, 'Hits@10': 0.501},
        'rotate':        {'MRR': 0.476, 'Hits@1': 0.428, 'Hits@3': 0.492, 'Hits@10': 0.571},
    },
}
# → MRR=0.3877  H@1=0.3732  H@3=0.3949  H@10=0.4118

# Label smoothing per model (TransE uses 0 since it's originally a margin model)
_LABEL_SMOOTHING = {
    'distmult_kge': 0.1,
    'transe': 0.0,
    'rotate': 0.1,
}

ALL_BENCHMARKS = ['fb15k-237', 'wn18rr']
ALL_MODELS = ['distmult_kge', 'transe', 'rotate']


# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(benchmark, model_name, config_name, runs, epochs, patience,
                   neg_batch_size, evaluate_every):
    """Train and evaluate one (benchmark, model) combination. Returns metrics dict."""

    label_smoothing = _LABEL_SMOOTHING.get(model_name, 0.1)
    loss_kwargs = dict(
        loss_fn='bce',
        label_smoothing=label_smoothing,
        alpha=0.25, gamma=3.0, alpha_adv=2.0,   # focal params, ignored when loss='bce'
    )
    # FB15k-237 and WN18RR are homogeneous: standard full-graph filtered ranking
    use_type_constrained = False

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

        train_index = torch.tensor(ds['train_index']).to(device)
        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}
        train_triplets = ds['train_triplets']
        val_triplets   = ds['val_triplets']
        test_triplets  = ds['test_triplets']
        train_val_triplets      = ds['train_val_triplets'].to(device)
        train_val_test_triplets = ds['train_val_test_triplets'].to(device)

        all_entities_arr = np.arange(ds['num_entities'])
        all_true_arr = train_val_test_triplets.cpu().numpy()
        reg_param = mp.get('regularization', 1e-5)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            all_params, lr=mp['learning_rate'], weight_decay=mp['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=mp.get('scheduler_gamma', 0.995)
        )

        best_val_loss = float('inf')
        last_improvement = 0
        best_state = None
        rng_batch = np.random.RandomState(seed)

        with trange(1, epochs + 1, desc=f'    {benchmark}/{model_name}') as pbar:
            for epoch in pbar:
                # Mini-batch positive subsampling (optional)
                if neg_batch_size > 0 and len(train_triplets) > neg_batch_size:
                    idx = rng_batch.choice(len(train_triplets), size=neg_batch_size, replace=False)
                    batch_triplets = train_triplets[idx]
                else:
                    batch_triplets = train_triplets

                neg_trips, neg_labels = negative_sampling_filtered(
                    batch_triplets, all_entities_arr, 1, all_true_arr, seed=seed + epoch
                )
                neg_trips, neg_labels = neg_trips.to(device), neg_labels.to(device)

                train_m = train_step(
                    encoder, decoder, optimizer, mp['grad_norm'], reg_param,
                    features, train_index, neg_trips, neg_labels,
                    **loss_kwargs,
                )
                scheduler.step()

                # Validation (sampled MRR — fast)
                if epoch % evaluate_every == 0:
                    val_neg, val_lab = negative_sampling_filtered(
                        val_triplets, all_entities_arr, 1, all_true_arr, seed=seed + 1000
                    )
                    val_neg, val_lab = val_neg.to(device), val_lab.to(device)

                    val_m = eval_step(
                        encoder, decoder, reg_param,
                        features, train_index, val_neg, val_lab, train_val_triplets,
                        **loss_kwargs,
                        eval_filtered=False,   # sampled during training for speed
                        use_type_constrained=use_type_constrained,
                    )

                    if val_m['Loss'] < best_val_loss:
                        best_val_loss = val_m['Loss']
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

        # Test: full filtered ranking (all entities)
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
        'KGE Link Prediction Benchmark Results',
        f'Generated : {ts}',
        f'Config    : {args.config}',
        f'Epochs    : {args.epochs}  |  Patience: {args.patience}  |  Runs: {args.runs}',
        f'Device    : {device}',
        '=' * 70,
    ]
    for benchmark in args.benchmarks:
        lines.append(f'\n{benchmark.upper()}')
        lines.append('-' * len(benchmark))
        for model in args.models:
            key = f'{benchmark}/{model}'
            if key not in results:
                lines.append(f'  {model:<16} [skipped / failed]')
                continue
            r = results[key]
            paper = PAPER_RESULTS.get(benchmark, {}).get(model)
            lines.append(_fmt_row(model, r['avg'], r['std'], paper))
    lines.append('\n' + '=' * 70)
    return '\n'.join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs('reports', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    json_path = f'reports/kge_lp_benchmark_{ts}.json'
    txt_path  = f'reports/kge_lp_benchmark_{ts}.txt'

    print(f'[i] Device: {device}')
    print(f'[i] Benchmarks: {args.benchmarks}')
    print(f'[i] Models    : {args.models}')
    print(f'[i] Config    : {args.config}')
    print(f'[i] Epochs    : {args.epochs}  Patience: {args.patience}  Runs: {args.runs}')
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
    parser = argparse.ArgumentParser(description='KGE Link Prediction Benchmark')
    parser.add_argument('--benchmarks', nargs='+', default=ALL_BENCHMARKS,
                        choices=ALL_BENCHMARKS,
                        help='Benchmarks to test (default: fb15k-237 wn18rr)')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='KGE models to test (default: distmult_kge transe rotate)')
    parser.add_argument('--config', type=str, default='lp-benchmark',
                        help='Config name from models_params.json (default: lp-benchmark = 200-dim)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs per experiment (default: 1)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max training epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience in epochs (default: 50)')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Validate every N epochs (default: 10)')
    parser.add_argument('--neg_batch_size', type=int, default=0,
                        help='Positive batch size before neg sampling. 0=full batch (default: 0)')
    args = parser.parse_args()
    main(args)
