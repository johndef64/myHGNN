"""
1-vs-All GNN Link Prediction Benchmark — corrected to match CompGCN paper protocol.

Six structural fixes over the original naive version (see comments for details):

  FIX 1 — Early stopping on val MRR instead of val BCE loss
    Val BCE loss decreases monotonically even when the model collapses (all logits
    → −∞ gives near-zero BCE on near-zero targets). The "best" checkpoint by BCE
    was the most-collapsed, worst-MRR model. The original CompGCN uses full-graph
    filtered val MRR for early stopping, which correctly selects the best model.

  FIX 2 — Encoder runs once per mini-batch (not once per epoch)
    The previous implementation ran the encoder once per epoch and accumulated
    gradients across batches with retain_graph=True. The original CompGCN runs
    zero_grad → encoder forward → loss → backward → step for every mini-batch.
    The per-epoch approach creates very different (unstable) gradient dynamics.

  FIX 3 — Multi-label 1-vs-All training grouped by unique (sub, rel) pairs
    The original CompGCN trains on unique (subject, relation) queries where the
    label is a multi-hot vector marking ALL valid tail entities for that query
    (from sr2o in data_loader.py). The previous code used individual triples with
    single-label targets, treating other valid answers as false negatives.
    Inverse queries (obj, rel+R) for head prediction are included, exactly as
    in the original sr2o construction.

  FIX 4 — Per-entity bias (critical for training stability)
    Original CompGCN: self.bias = Parameter(zeros(num_ent)); x += bias.expand_as(x).
    Without this, 1-vs-All BCE collapses: N−1 negative BCE terms always outweigh
    the positive term, pushing all logits toward −∞ (BCE→0, MRR→0 degenerate
    attractor). The per-entity bias provides entity-specific shortcuts that
    allow the model to maintain non-degenerate predictions during training.

  FIX 5 — Disable LayerNorm in CompGCN encoder
    The original CompGCN encoder has no normalization layers — only tanh
    activation between GCN layers, nothing after the last layer. Our encoder
    applies LayerNorm after every layer by default, which constrains embedding
    scales and makes it harder to escape the degenerate attractor in FIX 4.
    Disabled at runtime if the encoder has use_layer_norm=True.

  FIX 6 — Remove explicit L2 regularisation (l2=0.0 in original)
    Original CompGCN uses l2=0.0 (Adam without weight decay, no reg term).
    Our previous per-batch L2 effectively applied ~n_batches × reg per epoch,
    grossly over-regularising and shrinking embeddings toward zero.

Training protocol (CompGCN / Vashishth et al. 2020):
  • For each unique (h, r), score ALL N entities as tail candidates.
  • Multi-hot label: 1 for every valid tail in the training set.
  • Label smoothing ε=0.1: target = (1−ε)·y + ε/N  (matches data_loader.py).
  • BCE loss averaged over B×N elements, no gradient accumulation.
  • Inverse queries (t, r+R) handle head prediction symmetrically.
  • Early stopping: full-graph filtered val MRR (maximise), patience in epochs.

Usage:
  python benchmark_gnn_lp_1vsall.py --models compgcn --benchmarks wn18rr
  python benchmark_gnn_lp_1vsall.py --models compgcn rgcn --benchmarks wn18rr fb15k-237
  python benchmark_gnn_lp_1vsall.py --batch_size 512   # reduce if OOM
  python benchmark_gnn_lp_1vsall.py --config lp-benchmark-64  # 64-dim

Memory per mini-batch (B queries × N entities):
  WN18RR  (N=40K, dim=200, B=128): 128×40K×4B  ≈  20 MB/step  — very cheap
  FB15k-237 (N=14K, dim=200, B=128): 128×14K×4B ≈   7 MB/step  — trivial

Val MRR evaluation cost (full-graph filtered, every evaluate_every epochs):
  WN18RR  (3034 val triples × 40K entities × 2 modes): ~few seconds on GPU
  FB15k-237 (17535 val triples × 14K entities × 2 modes): ~10–20 s on GPU
"""

import os
import json
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from train_link_prediction import build_model, device
from src.datasets.lp_benchmarks import load_lp_benchmark
from src.utils import set_seed
from src.evaluation_metrics_filtered import evaluation_metrics_filtered_fullgraph

warnings.simplefilter('ignore')

BASE_SEED    = 42
LABEL_SMOOTH = 0.1   # ε — matches CompGCN paper and data_loader.py

# ─── Paper reference values ───────────────────────────────────────────────────
PAPER_RESULTS = {
    'fb15k-237': {
        'rgcn':    {'MRR': 0.249, 'Hits@1': 0.151, 'Hits@3': 0.264, 'Hits@10': 0.417},
        'compgcn': {'MRR': 0.355, 'Hits@1': 0.264, 'Hits@3': 0.390, 'Hits@10': 0.535},
    },
    'wn18rr': {
        'compgcn': {'MRR': 0.479, 'Hits@1': 0.443, 'Hits@3': 0.494, 'Hits@10': 0.546},
    },
}

ALL_BENCHMARKS = ['fb15k-237', 'wn18rr']
ALL_MODELS     = ['rgcn', 'rgat', 'compgcn']


# ─── Encoder kwargs for R-GAT ─────────────────────────────────────────────────

def _make_encoder_kwargs(model_name, train_index):
    """Sort edges by relation type and compute change-point tensor for R-GAT."""
    if model_name != 'rgat':
        return train_index, {}
    sort_idx    = train_index[:, 1].argsort()
    train_index = train_index[sort_idx]
    rel_ids     = train_index[:, 1]
    change_points = torch.cat([
        torch.tensor([0], device=train_index.device),
        (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
        torch.tensor([rel_ids.size(0)], device=train_index.device),
    ])
    return train_index, {'change_points': change_points}


# ─── FIX 3: Multi-label training data construction ────────────────────────────

def build_multilabel_data(train_triplets, num_original_rels):
    """
    Build multi-label 1-vs-All training data by grouping unique (sub, rel) pairs.

    Mirrors CompGCN's sr2o construction in data_loader.py:
      • Forward queries : (h, r)         → all valid tails   t  in training set
      • Inverse queries : (t, r + R)     → all valid heads   h  in training set
    where R = num_original_rels.

    Including inverses means head and tail prediction are handled symmetrically
    through a single 1-vs-All score-all-entities pass, exactly as in the original.

    Args:
        train_triplets:    LongTensor [M, 3] with columns (head, rel, tail).
                           Relation IDs are in [0, num_original_rels).
        num_original_rels: Number of original (non-inverse) relation types R.

    Returns:
        queries: LongTensor [K, 2]  — unique (sub, rel) pairs; K ≤ 2·M
        labels:  list[list[int]]    — labels[i] = all valid answer entity IDs
                                      for queries[i].  Variable-length; lives on CPU.
    """
    sr2o = defaultdict(list)

    for triple in train_triplets.tolist():
        h, r, t = triple[0], triple[1], triple[2]
        # Tail prediction: query (h, r) → answer t
        sr2o[(h, r)].append(t)
        # Head prediction via inverse relation: query (t, r+R) → answer h
        sr2o[(t, r + num_original_rels)].append(h)

    queries, labels = [], []
    for (sub, rel), objs in sr2o.items():
        queries.append([sub, rel])
        labels.append(objs)      # list of int — kept ragged on CPU

    return torch.tensor(queries, dtype=torch.long), labels


# ─── FIX 2: Per-batch training (encoder re-runs every mini-batch) ─────────────

def train_epoch_1vsall(encoder, decoder, optimizer, grad_norm, reg_param,
                       x_dict, edge_index, queries, labels, num_entities,
                       label_smoothing, batch_size, encoder_kwargs,
                       entity_bias=None):
    """
    One epoch of 1-vs-All training following the original CompGCN protocol.

    KEY CHANGES from the previous implementation:
      • The encoder runs once per mini-batch (zero_grad → forward → backward →
        step), NOT once per epoch with gradient accumulation / retain_graph.
        This matches run_epoch() in CompGCN/run.py and gives stable gradient
        dynamics that avoid the collapse observed previously.
      • Multi-label targets: ALL valid objects for a (sub, rel) query are marked
        as positive (multi-hot), not just a single triple's object.
      • Label smoothing applied as  target = (1-ε)·y + ε/N,
        identical to TrainDataset.__getitem__ in CompGCN/data_loader.py.
      • Regularisation removed (original CompGCN uses l2=0.0).
      • Per-entity bias added to scores before BCE loss — this is the critical
        fix: the original CompGCN has a Parameter(zeros(num_ent)) bias that
        provides entity-specific shortcuts preventing score collapse.

    Args:
        encoder, decoder   : model components (decoder has no learnable params
                             in DistMult — included only for grad-clip symmetry)
        optimizer          : Adam optimiser
        grad_norm          : max norm for gradient clipping
        reg_param          : λ for L2 regularisation (should be 0.0, kept for API
                             compatibility but not used in the loss)
        x_dict             : initial node feature dict (or None for learned init)
        edge_index         : message-passing graph [E, 3] (head, rel, tail)
        queries            : LongTensor [K, 2] (sub, rel) from build_multilabel_data
        labels             : list[list[int]] — valid answer entity IDs per query
        num_entities (N)   : total number of entities
        label_smoothing (ε): label-smoothing coefficient
        batch_size         : number of queries per mini-batch
        encoder_kwargs     : extra kwargs forwarded to encoder.forward()
        entity_bias        : Optional Parameter [N] — per-entity bias term added to
                             all scores (matches original CompGCN's bias parameter)

    Returns:
        avg_task_loss: average BCE task loss over all mini-batches (float)
    """
    encoder.train()
    dev = next(encoder.parameters()).device
    N   = num_entities

    # ── Shuffle queries at the start of each epoch ────────────────────────────
    n    = len(queries)
    perm = torch.randperm(n)
    queries_s = queries[perm].to(dev)         # (K, 2) on GPU
    labels_s  = [labels[i] for i in perm.tolist()]   # reordered list-of-lists

    n_batches       = max(1, (n + batch_size - 1) // batch_size)
    total_task_loss = 0.0

    for i in range(n_batches):
        start   = i * batch_size
        end     = min(start + batch_size, n)
        batch_q = queries_s[start:end]          # (B, 2)
        batch_l = labels_s[start:end]           # list of B obj-lists
        B       = batch_q.size(0)

        # ── Fresh encoder forward for this mini-batch ─────────────────────────
        # This is the key structural fix: encoder re-runs per batch so each
        # optimizer step sees an independent forward graph (no retain_graph).
        optimizer.zero_grad()
        node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)

        # ── DistMult 1-vs-All scores: (sub * rel) @ E^T  →  (B, N) ──────────
        sub_e  = node_emb[batch_q[:, 0]]    # (B, dim)
        r_e    = rel_emb[batch_q[:, 1]]     # (B, dim)
        scores = torch.mm(sub_e * r_e, node_emb.t())   # (B, N)

        # ── Per-entity bias (FIX 4) ───────────────────────────────────────────
        # Original CompGCN: self.bias = Parameter(zeros(num_ent)); x += bias.
        # Without this, BCE training collapses: all N−1 negative terms push
        # embeddings toward zero, reaching a degenerate all-negative attractor
        # (MRR≈0) that BCE cannot distinguish from a good model (BCE≈0 both).
        # The bias provides entity-specific shortcuts that prevent this collapse.
        if entity_bias is not None:
            scores = scores + entity_bias.unsqueeze(0)   # (B, N) + (1, N)

        # ── Multi-hot label with CompGCN label smoothing ──────────────────────
        # Vectorised scatter — no Python loop over B items.
        # Build flat (row, col) index tensors for every positive pair in
        # the batch, then fill in one shot with index assignment.
        # Smoothing: target = (1−ε)·y + ε/N  (data_loader.py line 31)
        rows = torch.cat([
            torch.full((len(pos),), b, dtype=torch.long, device=dev)
            for b, pos in enumerate(batch_l)
        ])
        cols = torch.cat([
            torch.tensor(pos, dtype=torch.long, device=dev)
            for pos in batch_l
        ])
        target = torch.full((B, N), label_smoothing / N, device=dev)
        target[rows, cols] = 1.0 - label_smoothing + label_smoothing / N

        # ── BCE loss (no explicit L2 reg — matches original CompGCN l2=0.0) ───
        task_loss = F.binary_cross_entropy_with_logits(scores, target)

        task_loss.backward()
        params_to_clip = list(encoder.parameters()) + list(decoder.parameters())
        if entity_bias is not None:
            params_to_clip.append(entity_bias)
        torch.nn.utils.clip_grad_norm_(params_to_clip, grad_norm)
        optimizer.step()

        total_task_loss += task_loss.item()

    return total_task_loss / n_batches



# ─── FIX 1: Val MRR for early stopping ────────────────────────────────────────

@torch.no_grad()
def val_mrr_1vsall(encoder, decoder, x_dict, edge_index, encoder_kwargs,
                   val_triplets, all_filter_triplets, num_entities,
                   entity_bias=None):
    """
    Compute full-graph filtered MRR on the validation set.

    This replaces val BCE loss as the early-stopping criterion, matching the
    original CompGCN's use of val_results['mrr'] in run.py (line 388).

    Val BCE loss was a flawed early-stopping signal: as all logits drift toward
    −∞ (degenerate collapse), BCE on near-zero targets also goes to ~0, so the
    model that minimised BCE was the most-collapsed model with the worst MRR.
    Full-graph filtered MRR correctly peaks at the model with the best ranking.

    Args:
        encoder, decoder         : model components
        x_dict                   : node features
        edge_index               : message-passing graph
        encoder_kwargs           : extra encoder.forward() kwargs
        val_triplets             : LongTensor [V, 3] validation triples
        all_filter_triplets      : LongTensor [*, 3] all known true triples
                                   (train+val+test) used for filtered ranking
        num_entities             : N

    Returns:
        mrr: float — mean reciprocal rank on the validation set (higher = better)
    """
    encoder.eval()
    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    dev = node_emb.device

    # Thin wrapper so evaluation_metrics_filtered_fullgraph can call .distmult()
    # The bias is applied per-candidate entity: in tail mode the candidates vary
    # in column 2 (head col 0 is constant); in head mode they vary in column 0.
    _bias = entity_bias  # captured in closure

    class _SW:
        def eval(self): pass
        def distmult(self, _emb, trips):
            base = decoder(node_emb, rel_emb, trips)
            if _bias is None:
                return base
            # Detect mode: if col 0 is constant → tail prediction (rank by col 2)
            if trips[:, 0].min() == trips[:, 0].max():
                return base + _bias[trips[:, 2]]
            else:  # head prediction (rank by col 0)
                return base + _bias[trips[:, 0]]

    if not isinstance(val_triplets, torch.Tensor):
        val_triplets = torch.tensor(val_triplets, dtype=torch.long)

    val_t     = val_triplets.to(dev)
    all_nodes = torch.arange(num_entities, device=dev)

    results = evaluation_metrics_filtered_fullgraph(
        _SW(), node_emb, all_filter_triplets, val_t,
        all_nodes, dev, hits_k=[1, 3, 10]
    )
    return results   # full dict: mrr, hits@1, hits@3, hits@10


# ─── Full-graph test evaluation (unchanged) ───────────────────────────────────

@torch.no_grad()
def test_fullgraph(encoder, decoder, x_dict, edge_index, encoder_kwargs,
                   test_triplets, train_val_test_triplets, num_entities,
                   entity_bias=None):
    """Filtered ranking against all entities — standard KG benchmark protocol."""
    encoder.eval()
    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    dev = node_emb.device

    _bias = entity_bias

    class _SW:
        def eval(self): pass
        def distmult(self, _emb, trips):
            base = decoder(node_emb, rel_emb, trips)
            if _bias is None:
                return base
            if trips[:, 0].min() == trips[:, 0].max():
                return base + _bias[trips[:, 2]]
            else:
                return base + _bias[trips[:, 0]]

    if not isinstance(test_triplets, torch.Tensor):
        test_triplets = torch.tensor(test_triplets, dtype=torch.long)

    test_t    = test_triplets.to(dev)
    all_nodes = torch.arange(num_entities, device=dev)

    return evaluation_metrics_filtered_fullgraph(
        _SW(), node_emb, train_val_test_triplets, test_t,
        all_nodes, dev, hits_k=[1, 3, 10]
    )


# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(benchmark, model_name, config_name, runs, epochs, patience,
                   evaluate_every, batch_size):
    """
    Run one (benchmark, model) experiment for `runs` independent seeds.

    Training protocol (matches CompGCN):
      1. Build multi-label 1-vs-All dataset (FIX 3).
      2. Per epoch: run train_epoch_1vsall with per-batch encoder (FIX 2).
      3. Every evaluate_every epochs: compute full-graph filtered val MRR (FIX 1).
      4. Checkpoint the epoch with highest val MRR; stop when no improvement
         for `patience` consecutive epochs.
      5. Reload best checkpoint, evaluate on test set.
    """
    all_run_metrics = []

    for run_i in range(runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'  [Run {run_i+1}/{runs}] seed={seed}')

        ds           = load_lp_benchmark(benchmark, root='dataset/', device=device)
        num_entities = ds['num_entities']

        # num_relations = 2·R + 1  (original + inverse + self-loop)
        # num_original_rels R is needed to build inverse queries for multi-label
        num_original_rels = (ds['num_relations'] - 1) // 2

        encoder, decoder, mp = build_model(
            model_name, ds, config_name, './src/models_params.json'
        )
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # FIX 5 — Disable LayerNorm in CompGCN encoder
        # The original CompGCN has no normalization between layers (only tanh
        # activation between layers, nothing after the last layer). Our encoder
        # has use_layer_norm=True by default, which constrains embedding scales
        # and makes it harder for the model to escape the degenerate attractor.
        if hasattr(encoder, 'use_layer_norm') and encoder.use_layer_norm:
            encoder.use_layer_norm = False
            print(f'  [i] LayerNorm disabled in {model_name} encoder '
                  f'(original CompGCN has no normalization)')

        train_index_raw         = torch.tensor(ds['train_index']).to(device)
        train_index, enc_kwargs = _make_encoder_kwargs(model_name, train_index_raw)

        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}

        # ── FIX 3: build multi-label grouped training data ────────────────────
        # Unique (sub, rel) queries with multi-hot labels, including inverse
        # queries for head prediction. This replaces iterating over raw triples.
        train_queries, train_labels = build_multilabel_data(
            ds['train_triplets'], num_original_rels
        )
        print(f'  [i] Multi-label queries: {len(train_queries)} unique (sub,rel) pairs '
              f'({len(ds["train_triplets"])} raw train triples, '
              f'{num_original_rels} original relations)')

        val_triplets            = ds['val_triplets']
        test_triplets           = ds['test_triplets']
        # all_filter_triplets used for filtered ranking (train+val+test)
        all_filter_triplets     = ds['train_val_test_triplets'].to(device)

        # FIX 4 — Per-entity bias (critical for training stability)
        # Original CompGCN: self.bias = Parameter(zeros(num_ent)); added to all
        # logits before sigmoid. Without this, BCE with N~40K entities collapses:
        # the N−1 negative terms always outweigh the positive term, pushing all
        # scores to −∞ (all entities get near-zero probability), producing
        # near-zero BCE but MRR≈0 (worst possible ranking).
        entity_bias = torch.nn.Parameter(torch.zeros(num_entities, device=device))

        # FIX 6 — No explicit L2 regularisation (original CompGCN: l2=0.0)
        # Our previous per-batch L2 effectively multiplied regularisation by
        # n_batches (~26 for WN18RR at batch_size=4096), grossly over-regularising.
        reg_param  = 0.0  # kept for API compatibility; not applied in loss

        # Adam: entity_bias trained alongside encoder and decoder parameters
        all_params = (list(encoder.parameters())
                      + list(decoder.parameters())
                      + [entity_bias])
        optimizer  = torch.optim.Adam(all_params, lr=mp['learning_rate'])
        scheduler  = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=mp.get('scheduler_gamma', 0.995)
        )

        # ── FIX 1: early stop on val MRR (maximise), not BCE (minimise) ───────
        best_val_mrr     = -1.0
        last_improvement = 0
        best_state       = None

        # Header for the per-evaluation log lines printed below the tqdm bar
        print(f'\n    {"Ep":>5}  {"LR":>8}  {"TrainLoss":>9}  '
              f'{"valMRR":>7}  {"H@1":>6}  {"H@3":>6}  {"H@10":>6}  '
              f'{"BestMRR":>7}  {"no-imp":>6}')
        print(f'    {"-"*5}  {"-"*8}  {"-"*9}  '
              f'{"-"*7}  {"-"*6}  {"-"*6}  {"-"*6}  '
              f'{"-"*7}  {"-"*6}')

        with trange(1, epochs + 1, desc=f'    {benchmark}/{model_name}',
                    leave=True) as pbar:
            for epoch in pbar:

                # FIX 2: encoder runs per mini-batch inside this call
                train_loss = train_epoch_1vsall(
                    encoder, decoder, optimizer, mp['grad_norm'], reg_param,
                    features, train_index, train_queries, train_labels,
                    num_entities, LABEL_SMOOTH, batch_size, enc_kwargs,
                    entity_bias=entity_bias,
                )

                # Scheduler steps once per epoch (after all mini-batch updates)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                if epoch % evaluate_every == 0:
                    # FIX 1: compute full-graph filtered val MRR + full metrics
                    val_res = val_mrr_1vsall(
                        encoder, decoder, features, train_index, enc_kwargs,
                        val_triplets, all_filter_triplets, num_entities,
                        entity_bias=entity_bias,
                    )
                    v_mrr   = val_res['mrr']
                    v_h1    = val_res['hits@1']
                    v_h3    = val_res['hits@3']
                    v_h10   = val_res['hits@10']
                    no_imp  = epoch - last_improvement

                    is_best = v_mrr > best_val_mrr
                    if is_best:
                        best_val_mrr     = v_mrr
                        last_improvement = epoch
                        no_imp           = 0
                        best_state = {
                            'encoder': {k: v.cpu().clone()
                                        for k, v in encoder.state_dict().items()},
                            'decoder': {k: v.cpu().clone()
                                        for k, v in decoder.state_dict().items()},
                            'entity_bias': entity_bias.data.cpu().clone(),
                        }

                    # One clear line per evaluation printed below the progress bar
                    marker = ' ★' if is_best else '  '
                    print(f'\r    {epoch:>5}  {current_lr:.2e}  {train_loss:>9.4f}  '
                          f'{v_mrr:>7.4f}  {v_h1:>6.4f}  {v_h3:>6.4f}  {v_h10:>6.4f}  '
                          f'{best_val_mrr:>7.4f}  {no_imp:>6}{marker}')

                    # tqdm postfix: compact summary always visible in the bar
                    pbar.set_postfix(
                        loss=f'{train_loss:.4f}',
                        MRR=f'{v_mrr:.4f}',
                        best=f'{best_val_mrr:.4f}',
                        H10=f'{v_h10:.4f}',
                    )

                    if not is_best and (epoch - last_improvement) >= patience:
                        print(f'\n    [i] Early stopping at epoch {epoch} '
                              f'(no val MRR improvement for {patience} epochs, '
                              f'best epoch={last_improvement})')
                        break
                else:
                    pbar.set_postfix(loss=f'{train_loss:.4f}', lr=f'{current_lr:.2e}')

        # ── Reload best checkpoint ─────────────────────────────────────────────
        if best_state is not None:
            encoder.load_state_dict(
                {k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict(
                {k: v.to(device) for k, v in best_state['decoder'].items()})
            entity_bias.data.copy_(best_state['entity_bias'].to(device))

        # ── Full filtered ranking on test set ──────────────────────────────────
        results = test_fullgraph(
            encoder, decoder, features, train_index, enc_kwargs,
            test_triplets, all_filter_triplets, num_entities,
            entity_bias=entity_bias,
        )

        run_result = {
            'MRR':    results['mrr'],
            'Hits@1': results['hits@1'],
            'Hits@3': results['hits@3'],
            'Hits@10':results['hits@10'],
        }
        print(f'    → MRR={run_result["MRR"]:.4f}  H@1={run_result["Hits@1"]:.4f}  '
              f'H@3={run_result["Hits@3"]:.4f}  H@10={run_result["Hits@10"]:.4f}')
        all_run_metrics.append(run_result)

    keys = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    avg  = {k: float(np.mean([m[k] for m in all_run_metrics])) for k in keys}
    std  = {k: float(np.std ([m[k] for m in all_run_metrics])) for k in keys}
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
    ts    = time.strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '=' * 70,
        'GNN Link Prediction Benchmark — 1-vs-All Training (fixed)',
        f'Generated  : {ts}',
        f'Config     : {args.config}',
        f'Epochs     : {args.epochs}  |  Patience: {args.patience}  |  Runs: {args.runs}',
        f'Batch size : {args.batch_size}  (unique (sub,rel) queries per mini-batch)',
        f'Device     : {device}',
        f'Loss       : 1-vs-All BCE + label smoothing {LABEL_SMOOTH} (CompGCN paper)',
        f'Early stop : full-graph filtered val MRR (maximise)',
        '=' * 70,
        '',
        'Paper references (Vashishth et al. 2020, CompGCN — ICLR, Table 3):',
        '  R-GCN+DistMult  (FB15k-237): MRR=0.249',
        '  CompGCN+DistMult (FB15k-237): MRR=0.355 | (WN18RR): MRR=0.479',
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
    ts        = time.strftime('%Y%m%d_%H%M%S')
    json_path = f'reports/gnn_lp_1vsall_{ts}.json'
    txt_path  = f'reports/gnn_lp_1vsall_{ts}.txt'

    print(f'[i] Device     : {device}')
    print(f'[i] Benchmarks : {args.benchmarks}')
    print(f'[i] Models     : {args.models}')
    print(f'[i] Config     : {args.config}')
    print(f'[i] Epochs     : {args.epochs}  Patience: {args.patience}  Runs: {args.runs}')
    print(f'[i] Batch size : {args.batch_size}  (unique (sub,rel) queries per batch)')
    print(f'[i] Eval every : every {args.evaluate_every} epochs  (full-graph val MRR)')
    print(f'[i] Loss       : 1-vs-All BCE + label smoothing {LABEL_SMOOTH}')
    print(f'[i] Early stop : full-graph filtered val MRR (maximise)')
    print()

    results = {}
    total   = len(args.benchmarks) * len(args.models)
    done    = 0

    for benchmark in args.benchmarks:
        for model in args.models:
            done += 1
            print(f'\n[{done}/{total}] {benchmark} / {model}')
            print('-' * 50)
            try:
                r = run_experiment(
                    benchmark      = benchmark,
                    model_name     = model,
                    config_name    = args.config,
                    runs           = args.runs,
                    epochs         = args.epochs,
                    patience       = args.patience,
                    evaluate_every = args.evaluate_every,
                    batch_size     = args.batch_size,
                )
                results[f'{benchmark}/{model}'] = r

                with open(json_path, 'w') as f:
                    json.dump({'args': vars(args), 'results': results}, f, indent=2)

            except Exception as e:
                import traceback
                print(f'  [ERROR] {e}')
                traceback.print_exc()
                results[f'{benchmark}/{model}'] = {'error': str(e)}

    report_txt = build_text_report(results, args)
    print('\n' + report_txt)

    with open(json_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)

    print(f'\n[i] Report saved to:\n    {json_path}\n    {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GNN Link Prediction Benchmark — 1-vs-All Training (fixed)'
    )
    parser.add_argument('--benchmarks', nargs='+', default=ALL_BENCHMARKS,
                        choices=['fb15k-237', 'wn18rr'],
                        help='Benchmarks (default: fb15k-237 wn18rr)')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='GNN models (default: rgcn rgat compgcn)')
    parser.add_argument('--config', type=str, default='lp-benchmark',
                        help='Config from models_params.json (default: lp-benchmark = 200-dim)')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stopping patience in epochs on val MRR (default: 200)')
    parser.add_argument('--evaluate_every', type=int, default=5,
                        help='Compute val MRR every N epochs (default: 5). '
                             'Val MRR is a full-graph filtered evaluation — '
                             'reduce to 10–20 for large datasets if too slow.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Unique (sub,rel) queries per mini-batch (default: 4096). '
                             'Each query scores all N entities: VRAM = B×N×4B. '
                             'WN18RR  B=4096: ~672 MB/step — fine on 24 GB GPU. '
                             'FB15k-237 B=4096: ~230 MB/step — fine. '
                             'Reduce to 1024/2048 only if OOM. '
                             'Larger B = fewer GNN passes per epoch = faster training.')
    args = parser.parse_args()
    main(args)


"""

batch_size	batches/epoch	stima s/epoch	500 epochs
128 (attuale)	809	~108s	~15h
512	~202	~27s	~3.8h
1024	~101	~13s	~1.8h
2048	~51	~7s	~1h
4096	~26	~4s	~0.5h
8192	~13	~2s	~0.25h
16384	~7	~1s	~0.125h
"""