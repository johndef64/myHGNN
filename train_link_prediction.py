"""
Train and evaluate heterogeneous GNN models for LINK PREDICTION on knowledge graphs.

Uses modular encoder/decoder architecture:
  - Encoders: CompGCN, R-GCN, R-GAT
  - Decoder: DistMult

Su Bio-KG la evaluation deve essere type-constrained (filtraggio solo contro entità del tipo corretto), altrimenti MRR e Hits@K saranno molto bassi a causa dell'enorme numero di entità negative. Su benchmark standard (FB15k-237, WN18RR) è invece più comune fare filtraggio completo (contro tutte le entità), ma è possibile scegliere anche il filtraggio type-constrained.

Training mode is selected automatically based on --loss:
  --loss bce   → 1-vs-All (CompGCN paper protocol): scores ALL entities per query,
                 multi-hot labels, entity bias, no weight decay, filtered val MRR
  --loss focal → negative sampling: focal loss + adversarial weighting (imbalanced datasets)
  --loss margin→ negative sampling: pairwise margin loss (TransE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BENCHMARKS  (1-vs-All BCE, full-graph filtered val/test MRR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # FB15k-237
  python train_link_prediction.py --benchmark fb15k-237 --model compgcn --config_name lp-benchmark --loss bce --label_smoothing 0.1 --early_stopping -e 500
  python train_link_prediction.py --benchmark fb15k-237 --model rgcn    --config_name lp-benchmark --loss bce --label_smoothing 0.1 --early_stopping -e 500

  # WN18RR
  python train_link_prediction.py --benchmark wn18rr --model compgcn --config_name lp-benchmark --loss bce --label_smoothing 0.1 --early_stopping -e 500
  python train_link_prediction.py --benchmark wn18rr --model rgcn    --config_name lp-benchmark --loss bce --label_smoothing 0.1 --early_stopping -e 500

  # ogbl-biokg  (type-constrained val/test MRR, large graph → bigger batch)
  python train_link_prediction.py --benchmark ogbl-biokg --model compgcn --config_name lp-benchmark --loss bce --label_smoothing 0.1 --neg_batch_size 512 --early_stopping -e 500
  python train_link_prediction.py --benchmark ogbl-biokg --model rgcn    --config_name lp-benchmark --loss bce --label_smoothing 0.1 --neg_batch_size 512 --early_stopping -e 500

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CUSTOM DATASETS  (negative sampling, type-constrained val/test MRR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # PathogenKG — focal loss (sbilanciato), task TARGET
  python train_link_prediction.py --model compgcn --task TARGET --loss focal --early_stopping -e 400
  python train_link_prediction.py --model rgcn    --task TARGET --loss focal --early_stopping -e 400
  python train_link_prediction.py --model compgcn --task TARGET --loss focal --runs 5 --early_stopping -e 400

  # PathogenKG — BCE + 1-vs-All (se vuoi replicare il protocollo benchmark)
  python train_link_prediction.py --model compgcn --task TARGET --loss bce --label_smoothing 0.1 --early_stopping -e 400

  # DRKG
  python train_link_prediction.py --model rgcn --tsv dataset/drkg/drkg_reduced.tsv --task Compound-Gene --loss focal --early_stopping -e 400

  # Hetionet
  python train_link_prediction.py --model rgcn --tsv dataset/hetionet/edges.tsv --task CcSE --loss focal --early_stopping -e 400

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KGE BASELINES  (pure embedding models, no GNN encoder)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # DistMult
  python train_link_prediction.py --benchmark fb15k-237  --model distmult_kge --config_name lp-benchmark --loss bce
  python train_link_prediction.py --benchmark ogbl-biokg --model distmult_kge --config_name lp-benchmark --loss bce

  # TransE  (margin loss — paper protocol)
  python train_link_prediction.py --benchmark fb15k-237 --model transe --config_name lp-benchmark --loss margin

  # RotatE
  python train_link_prediction.py --benchmark wn18rr    --model rotate --config_name lp-benchmark --loss bce
  python train_link_prediction.py --benchmark fb15k-237 --model rotate --config_name lp-benchmark --loss bce

  # Node2Vec + DistMult
  python train_link_prediction.py --model node2vec --task TARGET --loss bce

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
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.encoders import get_encoder
from src.decoders.distmult import DistMultDecoder
from src.decoders.transe import TransEDecoder
from src.decoders.rotate import RotatEDecoder
from src.datasets.kg_dataset import load_kg_dataset
from src.datasets.lp_benchmarks import load_lp_benchmark, LP_BENCHMARK_DATASETS
from src.utils import (
    set_seed, negative_sampling, negative_sampling_filtered,
    evaluation_metrics_sampled
)
from src.evaluation_metrics_filtered import (
    evaluation_metrics_filtered_fullgraph,
    evaluation_metrics_filtered_typeconstrained,
)

warnings.simplefilter(action='ignore')

BASE_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Loss ────────────────────────────────────────────────────────────────────

def focal_loss(inputs, targets, alpha=0.25, gamma=1.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = torch.exp(-bce)
    loss = alpha * (1 - p_t) ** gamma * bce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def compute_loss(decoder, node_emb, rel_emb, triplets, labels,
                 reg_param,
                 loss_fn='focal',
                 # focal-specific
                 alpha=0.25, gamma=1.0, alpha_adv=2.0,
                 # bce-specific
                 label_smoothing=0.0,
                 # margin-specific (TransE)
                 margin=1.0):
    """
    Unified loss dispatcher.

    Args:
        loss_fn: 'focal' | 'bce' | 'margin'
            - 'focal':  Focal Loss + adversarial hard-negative weighting.
                        Utile per dataset sbilanciati (PathogenKG, DRKG).
                        Parametri: alpha, gamma, alpha_adv.
            - 'bce':    BCE standard con label smoothing opzionale.
                        Come usato nel paper CompGCN (FB15k-237, WN18RR).
                        Parametro: label_smoothing (0.0 = nessuno, 0.1 = paper).
            - 'margin': Pairwise margin loss (TransE paper, Bordes 2013).
                        max(0, margin - score_pos + score_neg).
                        Richiede batch layout [pos | neg] da negative_sampling.
                        Parametro: margin (default 1.0).
        reg_param: coefficiente per la L2 regularization sugli embedding.

    Returns:
        (total_loss, scores): scores sono probabilità in [0, 1] pronte per AUROC/AUPRC.
    """
    raw_logits = decoder(node_emb, rel_emb, triplets)
    reg_loss = decoder.reg_loss(node_emb, rel_emb, triplets)

    if loss_fn == 'bce':
        # ── BCE con label smoothing (fedele al paper CompGCN) ──────────────
        if label_smoothing > 0.0:
            # Smoothing: sposta i target da {0,1} verso {ε/2, 1-ε/2}
            smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            smooth_labels = labels
        task_loss = F.binary_cross_entropy_with_logits(raw_logits, smooth_labels)

    elif loss_fn == 'focal':
        # ── Focal Loss + adversarial hard-negative mining ──────────────────
        probs = torch.sigmoid(raw_logits)
        fl_all = focal_loss(raw_logits, labels, alpha, gamma, reduction='none')

        pos_mask = labels.bool()
        neg_mask = ~pos_mask

        pos_loss = fl_all[pos_mask].mean() if pos_mask.any() else raw_logits.new_tensor(0.0)

        if neg_mask.any():
            neg_probs = probs[neg_mask]
            neg_weights = torch.softmax(neg_probs * alpha_adv, dim=0)
            neg_loss = (neg_weights * fl_all[neg_mask]).sum()
        else:
            neg_loss = raw_logits.new_tensor(0.0)

        task_loss = pos_loss + neg_loss

    elif loss_fn == 'margin':
        # ── Pairwise margin loss (TransE paper) ────────────────────────────
        # L = mean max(0, margin - score(pos) + score(neg))
        # Batch layout emesso da negative_sampling: [pos | neg] con rate negativi
        # per ogni positivo, quindi neg.size(0) / pos.size(0) = rate.
        pos_mask = labels.bool()
        pos_scores = raw_logits[pos_mask]    # [n_pos]
        neg_scores = raw_logits[~pos_mask]   # [n_pos * rate]
        n_pos = pos_scores.size(0)
        if n_pos == 0 or neg_scores.size(0) == 0:
            task_loss = raw_logits.new_tensor(0.0)
        else:
            rate = neg_scores.size(0) // n_pos
            neg_mat = neg_scores.view(n_pos, max(rate, 1))   # [n_pos, rate]
            task_loss = F.relu(margin - pos_scores.unsqueeze(1) + neg_mat).mean()

    else:
        raise ValueError(f"loss_fn must be 'bce', 'focal' or 'margin', got '{loss_fn}'")

    total_loss = task_loss + reg_param * reg_loss
    scores = torch.sigmoid(raw_logits)   # un solo sigmoid, su raw_logits
    return total_loss, scores


# ─── Model factory ───────────────────────────────────────────────────────────

_KGE_MODELS = {'distmult_kge', 'transe', 'rotate', 'node2vec'}
_KGE_DECODER_MAP = {
    'distmult_kge': DistMultDecoder,
    'transe': TransEDecoder,
    'rotate': RotatEDecoder,
    'node2vec': DistMultDecoder,
}


def build_model(model_name, dataset_info, config_name, models_params_path):
    with open(models_params_path, 'r') as f:
        models_params = json.load(f)

    if config_name not in models_params:
        print(f"[!] Config '{config_name}' not found, using 'default'")
        config_name = "default"

    if model_name not in models_params[config_name]:
        # Fall back to 'default' params for this model (covers KGE models
        # not explicitly listed in every config)
        if model_name in models_params.get('default', {}):
            print(f"[!] Model '{model_name}' not in config '{config_name}', using 'default' params")
            mp = models_params['default'][model_name]
        else:
            raise KeyError(f"Model '{model_name}' not found in config '{config_name}' or 'default'")
    else:
        mp = models_params[config_name][model_name]

    print(f"[i] Model params ({config_name}): {mp}")

    conv_hidden = {f'layer_{x}': mp[f'layer_{x}'] for x in range(mp['conv_layer_num'])}

    EncoderClass = get_encoder(model_name)

    if model_name == 'compgcn':
        encoder = EncoderClass(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=dataset_info['num_relations'],
            conv_num_layers=mp['conv_layer_num'],
            opn=mp['opn'],
            dropout=mp['dropout'],
            device=device,
        )
    elif model_name in _KGE_MODELS:
        encoder = EncoderClass(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=dataset_info['num_relations'],
            conv_num_layers=mp['conv_layer_num'],
            device=device,
            dropout=mp.get('dropout', 0.0),
            # TransE / RotatE
            gamma=mp.get('gamma', 12.0),
            p_norm=mp.get('p_norm', 1),
            # Node2Vec
            walk_length=mp.get('walk_length', 20),
            context_size=mp.get('context_size', 10),
            walks_per_node=mp.get('walks_per_node', 10),
            p=mp.get('p', 1.0),
            q=mp.get('q', 1.0),
            pretrain_epochs=mp.get('pretrain_epochs', 5),
        )
        DecoderClass = _KGE_DECODER_MAP[model_name]
        decoder = DecoderClass(gamma=mp.get('gamma', 12.0), p_norm=mp.get('p_norm', 1))
        return encoder, decoder, mp
    else:  # rgcn, rgat
        encoder = EncoderClass(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_hidden_channels_dict=mp['mlp_out_layer'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=dataset_info['num_relations'] + 1,
            conv_num_layers=mp['conv_layer_num'],
            num_bases=mp['num_bases'],
            activation_function=F.relu,
            device=device,
        )

    decoder = DistMultDecoder()

    return encoder, decoder, mp


# ─── Training step ───────────────────────────────────────────────────────────

def train_step(encoder, decoder, optimizer, grad_norm, reg_param,
               x_dict, edge_index, triplets, labels,
               loss_fn, alpha, gamma, alpha_adv, label_smoothing, margin=1.0,
               **encoder_kwargs):
    encoder.train()
    optimizer.zero_grad()

    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    loss, scores = compute_loss(
        decoder, node_emb, rel_emb, triplets, labels, reg_param,
        loss_fn=loss_fn,
        alpha=alpha, gamma=gamma, alpha_adv=alpha_adv,
        label_smoothing=label_smoothing, margin=margin,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()), grad_norm
    )
    optimizer.step()

    return {
        'Loss': loss.item(),
        'Auroc': binary_auroc(scores, labels).item(),
        'Auprc': binary_auprc(scores, labels).item(),
    }


# ─── Evaluation step ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(encoder, decoder, reg_param, x_dict, edge_index,
              triplets, labels, train_val_triplets,
              loss_fn, alpha, gamma, alpha_adv, label_smoothing, margin=1.0,
              eval_filtered=False, all_target_triplets=None,
              num_entities=None, use_type_constrained=True,
              **encoder_kwargs):
    """
    evaluation fatta durante il training (val) o alla fine (test).
    """
    encoder.eval()

    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    loss, scores = compute_loss(
        decoder, node_emb, rel_emb, triplets, labels, reg_param,
        loss_fn=loss_fn,
        alpha=alpha, gamma=gamma, alpha_adv=alpha_adv,
        label_smoothing=label_smoothing, margin=margin,
    )

    metrics = {
        'Loss': loss.item(),
        'Auroc': binary_auroc(scores, labels).item(),
        'Auprc': binary_auprc(scores, labels).item(),
    }

    if eval_filtered and all_target_triplets is not None and num_entities is not None:
        pos_mask = labels.bool()
        test_positives = triplets[pos_mask]
        all_graph_nodes = torch.arange(num_entities, device=triplets.device)

        if use_type_constrained:
            filtered = evaluation_metrics_filtered_typeconstrained(
                _ScoringWrapper(decoder, node_emb, rel_emb),
                node_emb, all_target_triplets, test_positives,
                all_graph_nodes, triplets.device, hits_k=[1, 3, 10]
            )
        else:
            filtered = evaluation_metrics_filtered_fullgraph(
                _ScoringWrapper(decoder, node_emb, rel_emb),
                node_emb, all_target_triplets, test_positives,
                all_graph_nodes, triplets.device, hits_k=[1, 3, 10]
            )
        metrics['MRR'] = filtered['mrr']
        metrics['Hits@'] = {k: filtered[f'hits@{k}'] for k in [1, 3, 10]}
    else:
        NUM_GENERATE = 200
        pos_mask = labels.bool()
        pos_triplets = triplets[pos_mask]
        mrr, hits = evaluation_metrics_sampled(
            _ScoringWrapper(decoder, node_emb, rel_emb),
            node_emb, train_val_triplets, pos_triplets, NUM_GENERATE, 0,
            hits=[1, 3, 10]
        )
        metrics['MRR'] = mrr
        metrics['Hits@'] = hits

    return metrics


class _ScoringWrapper:
    """Wraps encoder+decoder to match the interface expected by evaluation_metrics_*."""
    def __init__(self, decoder, node_emb, rel_emb):
        self.decoder = decoder
        self.node_emb = node_emb
        self.rel_emb = rel_emb

    def eval(self):
        pass

    def distmult(self, embedding, triplets):
        return self.decoder(self.node_emb, self.rel_emb, triplets)


# ─── 1-vs-All training helpers (CompGCN paper protocol) ──────────────────────

def build_multilabel_data(train_triplets, num_original_rels):
    """
    Group unique (sub, rel) pairs with multi-hot labels.
    Mirrors CompGCN sr2o: forward queries (h,r)→t + inverse queries (t,r+R)→h.
    """
    sr2o = defaultdict(list)
    for triple in train_triplets.tolist():
        h, r, t = triple[0], triple[1], triple[2]
        sr2o[(h, r)].append(t)
        sr2o[(t, r + num_original_rels)].append(h)
    queries, labels = [], []
    for (sub, rel), objs in sr2o.items():
        queries.append([sub, rel])
        labels.append(objs)
    return torch.tensor(queries, dtype=torch.long), labels


def train_epoch_1vsall(encoder, decoder, optimizer, grad_norm,
                       x_dict, edge_index, queries, labels, num_entities,
                       label_smoothing, batch_size, encoder_kwargs,
                       entity_bias=None):
    """
    One epoch of 1-vs-All training: encoder re-runs per mini-batch.

    For each (sub, rel) query, scores ALL N entities as tail candidates.
    Multi-hot BCE with label smoothing (CompGCN paper).
    No explicit L2 reg (original CompGCN uses l2=0.0).
    """
    encoder.train()
    dev = next(encoder.parameters()).device
    N   = num_entities

    n    = len(queries)
    perm = torch.randperm(n)
    queries_s = queries[perm].to(dev)
    labels_s  = [labels[i] for i in perm.tolist()]

    n_batches       = max(1, (n + batch_size - 1) // batch_size)
    total_task_loss = 0.0

    for i in range(n_batches):
        start   = i * batch_size
        end     = min(start + batch_size, n)
        batch_q = queries_s[start:end]      # (B, 2)
        batch_l = labels_s[start:end]       # list of B obj-lists
        B       = batch_q.size(0)

        optimizer.zero_grad()
        node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)

        sub_e  = node_emb[batch_q[:, 0]]    # (B, dim)
        r_e    = rel_emb[batch_q[:, 1]]     # (B, dim)
        scores = torch.mm(sub_e * r_e, node_emb.t())   # (B, N)

        if entity_bias is not None:
            scores = scores + entity_bias.unsqueeze(0)  # (B, N)

        # Multi-hot label with label smoothing: target = (1-ε)·y + ε/N
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

        task_loss = F.binary_cross_entropy_with_logits(scores, target)
        task_loss.backward()

        params_to_clip = list(encoder.parameters()) + list(decoder.parameters())
        if entity_bias is not None:
            params_to_clip.append(entity_bias)
        torch.nn.utils.clip_grad_norm_(params_to_clip, grad_norm)
        optimizer.step()

        total_task_loss += task_loss.item()

    return total_task_loss / n_batches


@torch.no_grad()
def _val_mrr_1vsall(encoder, decoder, x_dict, edge_index, encoder_kwargs,
                    val_triplets, all_filter_triplets, num_entities,
                    use_type_constrained, entity_bias=None):
    """
    Full filtered val MRR for 1-vs-All early stopping.
    Uses full-graph filtering for standard benchmarks (fb15k-237, wn18rr)
    and type-constrained filtering for biokg / custom datasets.
    """
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
            # tail prediction: col 0 constant → add bias on col 2 (tail)
            if trips[:, 0].min() == trips[:, 0].max():
                return base + _bias[trips[:, 2]]
            else:  # head prediction
                return base + _bias[trips[:, 0]]

    if not isinstance(val_triplets, torch.Tensor):
        val_triplets = torch.tensor(val_triplets, dtype=torch.long)

    val_t     = val_triplets.to(dev)
    all_nodes = torch.arange(num_entities, device=dev)

    if use_type_constrained:
        results = evaluation_metrics_filtered_typeconstrained(
            _SW(), node_emb, all_filter_triplets, val_t,
            all_nodes, dev, hits_k=[1, 3, 10]
        )
    else:
        results = evaluation_metrics_filtered_fullgraph(
            _SW(), node_emb, all_filter_triplets, val_t,
            all_nodes, dev, hits_k=[1, 3, 10]
        )
    return results   # dict: mrr, hits@1, hits@3, hits@10


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):

    is_benchmark = args.benchmark is not None
    # fefinizhe quale funeionze di eval usare durante il training: se è un benchmark standard, conviene usare la versione sampled (più veloce, e più allineata alla evaluation finale filtrata); se è un dataset custom, lascio scegliere se usare eval filtrata (type-constrained) o no (di default eval non filtrata, per velocità).
    if is_benchmark:
        print(f'[i] Benchmark: {args.benchmark}')
        if args.config_name == 'pathogen31-64':
            args.config_name = 'lp-benchmark'
        # CRITICAL: val MRR must match the final evaluation type so the model
        # learns the correct ranking objective. Sampled-200 MRR during training
        # while evaluating with full-graph filtered MRR at test time creates a
        # mismatch: the model is never optimised for the actual metric.
        eval_filtered_val = True
        # ogbl-biokg: type-constrained (too large for full-graph ranking)
        # fb15k-237, wn18rr: full-graph (standard literature protocol)
        use_type_constrained = (args.benchmark == 'ogbl-biokg')
        print(f'[i] Benchmark mode: using {"type-constrained" if use_type_constrained else "full-graph"} '
              f'filtered val MRR during training (matches test evaluation).')
    else:
        print(f'[i] Dataset: {args.tsv}')
        eval_filtered_val = args.eval_filtered
        use_type_constrained = True  # KG eterogenei custom: sempre type-constrained

    print(f'[i] Test evaluation: {"type-constrained" if use_type_constrained else "full-graph"} filtered ranking')

    # Auto-switch loss for TransE: margin loss is the correct choice (paper Bordes 2013)
    if args.loss == 'auto':
        args.loss = 'margin' if args.model == 'transe' else 'focal'
        print(f'[i] Auto loss → {args.loss}')
    elif args.model == 'transe' and args.loss in ('focal', 'bce'):
        print(f'[!] TransE with {args.loss} loss: suboptimal (designed for margin loss). '
              f'Consider --loss margin.')

    if args.loss == 'bce':
        print(f"[i] Loss: bce (label_smoothing={args.label_smoothing})")
    elif args.loss == 'margin':
        print(f"[i] Loss: margin (margin={args.margin})")
    else:
        print(f"[i] Loss: focal (alpha={args.alpha}, gamma={args.gamma}, alpha_adv={args.alpha_adv})")

    models_params_path = './src/models_params.json'
    all_run_metrics = []

    neg_sampler = negative_sampling_filtered if args.negative_sampling == 'filtered' else negative_sampling
    print(f"[i] Negative sampling: {args.negative_sampling}")
    print(f"[i] Evaluation: {'filtered' if args.eval_filtered else 'legacy'}")

    # Kwargs fissi per compute_loss — passati invariati a train_step e eval_step
    loss_kwargs = dict(
        loss_fn=args.loss,
        alpha=args.alpha,
        gamma=args.gamma,
        alpha_adv=args.alpha_adv,
        label_smoothing=args.label_smoothing,
        margin=args.margin,
    )

    for run_i in range(args.runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'\n{"="*60}\n[i] Run {run_i}/{args.runs} | seed={seed}')

        # Load dataset
        if is_benchmark:
            ds = load_lp_benchmark(args.benchmark, root='dataset/', device=device)
        else:
            ds = load_kg_dataset(
                args.tsv, args.task, args.validation_size, args.test_size,
                args.oversample_rate, args.undersample_rate, seed, args.quiet, device
            )

        # Build model
        encoder, decoder, mp = build_model(
            args.model, ds, args.config_name, models_params_path
        )
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Move data to device
        train_index = torch.tensor(ds['train_index']).to(device)
        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}
        train_triplets = ds['train_triplets']
        val_triplets = ds['val_triplets']
        test_triplets = ds['test_triplets']
        train_val_triplets = ds['train_val_triplets'].to(device)
        train_val_test_triplets = ds['train_val_test_triplets'].to(device)

        all_entities_arr = np.arange(ds['num_entities'])
        all_true_arr = train_val_test_triplets.cpu().numpy()

        # Encoder-specific kwargs
        encoder_kwargs = {}
        if args.model == 'rgat':
            rel_ids = train_index[:, 1]
            encoder_kwargs['change_points'] = torch.cat([
                torch.tensor([0], device=device),
                (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
                torch.tensor([rel_ids.size(0)], device=device)
            ])

        # ── 1-vs-All mode: BCE loss scores ALL entities per (sub, rel) query ──
        # This is the CompGCN paper protocol. Requires:
        #   - Multi-hot labels (all valid tails for a query marked as positive)
        #   - Per-entity bias (prevents BCE collapse toward all-negative attractor)
        #   - No explicit L2 reg (original CompGCN l2=0.0)
        #   - LayerNorm disabled in CompGCN encoder
        # We use 1-vs-All when training with BCE loss (benchmarks and any custom
        # dataset using --loss bce). Focal/margin keep the negative-sampling path.
        use_1vsall = (args.loss == 'bce')

        if use_1vsall:
            # FIX 5: Disable LayerNorm in CompGCN (original has none)
            if hasattr(encoder, 'use_layer_norm') and encoder.use_layer_norm:
                encoder.use_layer_norm = False
                print(f'  [i] LayerNorm disabled in {args.model} encoder '
                      f'(1-vs-All BCE mode, matches original CompGCN)')

            # FIX 4: Per-entity bias — prevents score collapse in 1-vs-All BCE
            entity_bias = torch.nn.Parameter(torch.zeros(ds['num_entities'], device=device))

            # FIX 3: Build multi-label grouped training data
            num_original_rels = (ds['num_relations'] - 1) // 2
            train_queries, train_labels = build_multilabel_data(
                train_triplets, num_original_rels
            )
            print(f'  [i] Multi-label queries: {len(train_queries)} unique (sub,rel) pairs '
                  f'({len(train_triplets)} raw train triples, {num_original_rels} original relations)')

            # FIX 6: No weight decay — original CompGCN uses Adam, l2=0.0
            all_params = (list(encoder.parameters()) + list(decoder.parameters())
                          + [entity_bias])
            optimizer = torch.optim.Adam(all_params, lr=mp['learning_rate'])
        else:
            entity_bias = None
            all_params = list(encoder.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=mp['learning_rate'],
                                          weight_decay=mp['weight_decay'])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=mp['scheduler_gamma'])

        # Training loop
        best_val_loss = float('inf')
        best_val_mrr  = -1.0
        last_improvement = 0
        best_state = None
        save_path = None

        if not args.dry_run:
            ts = time.strftime('%Y%m%d_%H%M%S')
            if is_benchmark:
                task_clean = args.benchmark.replace('-', '_')
                ds_name = args.benchmark
            else:
                task_clean = args.task.lower().replace('-', '_').replace(',', '_')
                ds_name = os.path.splitext(os.path.basename(args.tsv))[0]
            folder = os.path.join('models', f'{task_clean}_{ds_name}_{ts}')
            os.makedirs(folder, exist_ok=True)
            save_path = os.path.join(folder, f'{args.model}_run{run_i}.pt')
            with open(os.path.join(folder, 'params.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)

        val_mrr  = 0.0
        val_h1   = 0.0
        val_h3   = 0.0
        val_h10  = 0.0
        val_loss = 0.0

        rng_batch = np.random.RandomState(seed)
        n_train = len(train_triplets)
        use_minibatch = args.neg_batch_size > 0 and n_train > args.neg_batch_size
        steps_per_epoch = (n_train + args.neg_batch_size - 1) // args.neg_batch_size if use_minibatch else 1

        # ── Table header (printed once before the tqdm bar) ───────────────────
        eval_label = 'type-constr' if use_type_constrained else 'full-graph'
        print(f'\n  {"Ep":>5}  {"LR":>8}  {"TrainLoss":>9}  '
              f'{"valMRR":>7}  {"H@1":>6}  {"H@3":>6}  {"H@10":>6}  '
              f'{"BestMRR":>7}  {"no-imp":>6}  [{eval_label}]')
        print(f'  {"-"*5}  {"-"*8}  {"-"*9}  '
              f'{"-"*7}  {"-"*6}  {"-"*6}  {"-"*6}  '
              f'{"-"*7}  {"-"*6}')

        with trange(1, args.epochs + 1, desc=f'  run {run_i+1}/{args.runs}',
                    leave=True) as pbar:
            for epoch in pbar:

                if use_1vsall:
                    # FIX 2: encoder re-runs per mini-batch (no retain_graph)
                    batch_size_1vsall = args.neg_batch_size if args.neg_batch_size > 0 else 128
                    train_loss = train_epoch_1vsall(
                        encoder, decoder, optimizer, mp['grad_norm'],
                        features, train_index, train_queries, train_labels,
                        ds['num_entities'], args.label_smoothing,
                        batch_size_1vsall, encoder_kwargs,
                        entity_bias=entity_bias,
                    )
                    train_m = {'Loss': train_loss}
                elif use_minibatch:
                    perm = rng_batch.permutation(n_train)
                    train_m = None
                    for step in range(steps_per_epoch):
                        start = step * args.neg_batch_size
                        end   = min(start + args.neg_batch_size, n_train)
                        batch_triplets = train_triplets[perm[start:end]]

                        if args.negative_sampling == 'filtered':
                            neg_trips, neg_labels = neg_sampler(
                                batch_triplets, all_entities_arr, args.negative_rate,
                                all_true_arr, seed=seed + epoch * steps_per_epoch + step
                            )
                        else:
                            neg_trips, neg_labels = neg_sampler(batch_triplets, args.negative_rate)
                        neg_trips, neg_labels = neg_trips.to(device), neg_labels.to(device)

                        train_m = train_step(
                            encoder, decoder, optimizer, mp['grad_norm'], mp['regularization'],
                            features, train_index, neg_trips, neg_labels,
                            **loss_kwargs, **encoder_kwargs
                        )
                else:
                    if args.negative_sampling == 'filtered':
                        neg_trips, neg_labels = neg_sampler(
                            train_triplets, all_entities_arr, args.negative_rate,
                            all_true_arr, seed=seed + epoch
                        )
                    else:
                        neg_trips, neg_labels = neg_sampler(train_triplets, args.negative_rate)
                    neg_trips, neg_labels = neg_trips.to(device), neg_labels.to(device)

                    train_m = train_step(
                        encoder, decoder, optimizer, mp['grad_norm'], mp['regularization'],
                        features, train_index, neg_trips, neg_labels,
                        **loss_kwargs, **encoder_kwargs
                    )

                scheduler.step()  # once per epoch
                current_lr = optimizer.param_groups[0]['lr']

                # Validate
                if epoch % args.evaluate_every == 0:
                    if use_1vsall:
                        val_res = _val_mrr_1vsall(
                            encoder, decoder, features, train_index, encoder_kwargs,
                            val_triplets, train_val_test_triplets,
                            ds['num_entities'], use_type_constrained,
                            entity_bias=entity_bias,
                        )
                        val_mrr  = val_res['mrr']
                        val_h1   = val_res['hits@1']
                        val_h3   = val_res['hits@3']
                        val_h10  = val_res['hits@10']
                        val_loss = 0.0
                    else:
                        if args.negative_sampling == 'filtered':
                            val_neg, val_lab = neg_sampler(
                                val_triplets, all_entities_arr, args.negative_rate,
                                all_true_arr, seed=seed + 1000
                            )
                        else:
                            val_neg, val_lab = neg_sampler(val_triplets, args.negative_rate)
                        val_neg, val_lab = val_neg.to(device), val_lab.to(device)

                        val_metrics = eval_step(
                            encoder, decoder, mp['regularization'],
                            features, train_index, val_neg, val_lab, train_val_triplets,
                            **loss_kwargs,
                            eval_filtered=eval_filtered_val,
                            all_target_triplets=train_val_test_triplets if eval_filtered_val else None,
                            num_entities=ds['num_entities'] if eval_filtered_val else None,
                            use_type_constrained=use_type_constrained,
                            **encoder_kwargs
                        )
                        val_mrr  = val_metrics.get('MRR', 0)
                        val_h1   = val_metrics.get('Hits@', {}).get(1, 0)
                        val_h3   = val_metrics.get('Hits@', {}).get(3, 0)
                        val_h10  = val_metrics.get('Hits@', {}).get(10, 0)
                        val_loss = val_metrics['Loss']

                    # Early stopping: on val MRR when filtered eval, on val loss otherwise
                    if eval_filtered_val:
                        val_improved = val_mrr > best_val_mrr
                    else:
                        val_improved = val_loss < best_val_loss - args.min_delta

                    no_imp  = 0 if val_improved else epoch - last_improvement
                    marker  = ' ★' if val_improved else '  '

                    if val_improved:
                        best_val_loss = val_loss
                        best_val_mrr  = val_mrr
                        last_improvement = epoch
                        best_state = {
                            'encoder': {k: v.cpu().clone()
                                        for k, v in encoder.state_dict().items()},
                            'decoder': {k: v.cpu().clone()
                                        for k, v in decoder.state_dict().items()},
                        }
                        if entity_bias is not None:
                            best_state['entity_bias'] = entity_bias.data.cpu().clone()
                        if save_path:
                            torch.save(best_state, save_path)

                    # One line per evaluation, printed below the tqdm bar
                    print(f'\r  {epoch:>5}  {current_lr:.2e}  {train_m["Loss"]:>9.4f}  '
                          f'{val_mrr:>7.4f}  {val_h1:>6.4f}  {val_h3:>6.4f}  {val_h10:>6.4f}  '
                          f'{best_val_mrr:>7.4f}  {no_imp:>6}{marker}')

                    pbar.set_postfix(
                        loss=f'{train_m["Loss"]:.4f}',
                        MRR=f'{val_mrr:.4f}',
                        best=f'{best_val_mrr:.4f}',
                        H10=f'{val_h10:.4f}',
                    )

                    if not val_improved and args.early_stopping and no_imp >= args.patience:
                        print(f'\n  [i] Early stopping at epoch {epoch} '
                              f'(no improvement for {args.patience} epochs, '
                              f'best epoch={last_improvement})')
                        break
                else:
                    pbar.set_postfix(loss=f'{train_m["Loss"]:.4f}', lr=f'{current_lr:.2e}')

        # ── Reload best checkpoint ─────────────────────────────────────────────
        if best_state is not None:
            encoder.load_state_dict(
                {k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict(
                {k: v.to(device) for k, v in best_state['decoder'].items()})
            if entity_bias is not None and 'entity_bias' in best_state:
                entity_bias.data.copy_(best_state['entity_bias'].to(device))
        elif save_path and os.path.exists(save_path):
            ckpt = torch.load(save_path)
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['decoder'])
            if entity_bias is not None and 'entity_bias' in ckpt:
                entity_bias.data.copy_(ckpt['entity_bias'].to(device))

        if use_1vsall:
            # Test: same scoring as val — filtered full-graph or type-constrained
            test_res = _val_mrr_1vsall(
                encoder, decoder, features, train_index, encoder_kwargs,
                test_triplets, train_val_test_triplets,
                ds['num_entities'], use_type_constrained,
                entity_bias=entity_bias,
            )
            test_m = {
                'Auroc': 0.0, 'Auprc': 0.0,
                'MRR':    test_res['mrr'],
                'Hits@': {1: test_res['hits@1'], 3: test_res['hits@3'],
                          10: test_res['hits@10']},
            }
        else:
            if args.negative_sampling == 'filtered':
                test_neg, test_lab = neg_sampler(
                    test_triplets, all_entities_arr, args.negative_rate,
                    all_true_arr, seed=seed + 2000
                )
            else:
                test_neg, test_lab = neg_sampler(test_triplets, args.negative_rate)
            test_neg, test_lab = test_neg.to(device), test_lab.to(device)

            test_m = eval_step(
                encoder, decoder, mp['regularization'],
                features, train_index, test_neg, test_lab, train_val_test_triplets,
                **loss_kwargs,
                eval_filtered=args.eval_filtered,
                all_target_triplets=train_val_test_triplets if args.eval_filtered else None,
                num_entities=ds['num_entities'] if args.eval_filtered else None,
                use_type_constrained=use_type_constrained,
                **encoder_kwargs
            )

        print(f"Run {run_i} | AUROC: {test_m['Auroc']:.3f}, AUPRC: {test_m['Auprc']:.3f}, "
              f"MRR: {test_m['MRR']:.3f}, Hits@: {test_m['Hits@']}")

        all_run_metrics.append({
            'Auroc': test_m['Auroc'], 'Auprc': test_m['Auprc'],
            'MRR': test_m['MRR'],
            'Hits@1': test_m['Hits@'][1], 'Hits@3': test_m['Hits@'][3],
            'Hits@10': test_m['Hits@'][10],
        })



    # Summary
    if all_run_metrics:
        keys = ['Auroc', 'Auprc', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']
        avg = {k: float(np.mean([m[k] for m in all_run_metrics])) for k in keys}
        std = {k: float(np.std([m[k] for m in all_run_metrics])) for k in keys}
        print(f"\n{'='*60}")
        print("SUMMARY:")
        for k in keys:
            print(f"  {k}: {avg[k]:.4f} ± {std[k]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link Prediction on Heterogeneous KGs')
    parser.add_argument('--tsv', type=str, default='dataset/PathogenKG_n31_core.tsv.zip')
    parser.add_argument('--benchmark', type=str, default=None, choices=LP_BENCHMARK_DATASETS,
                        help='Use a standard KG benchmark instead of --tsv. '
                             'Choices: fb15k-237, wn18rr, ogbl-biokg')
    parser.add_argument('-m', '--model', type=str, default='compgcn',
                        choices=['rgcn', 'rgat', 'compgcn',
                                 'distmult_kge', 'transe', 'rotate', 'node2vec'])
    parser.add_argument('--task', type=str, default='TARGET')
    parser.add_argument('--config_name', type=str, default='pathogen31-64')
    parser.add_argument('-r', '--runs', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=400)
    parser.add_argument('-p', '--patience', type=int, default=20)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--validation_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--evaluate_every', type=int, default=5)
    parser.add_argument('--negative_sampling', type=str, default='filtered')
    parser.add_argument('--negative_rate', type=float, default=1)
    parser.add_argument('--neg_batch_size', type=int, default=0,
                        help='Mini-batch size for positive triples before negative sampling. '
                             '0 = disabled (full batch). Use e.g. 4096 or 2048 for large benchmarks.')
    parser.add_argument('--oversample_rate', type=int, default=5)
    parser.add_argument('--undersample_rate', type=float, default=0.5)

    # ── Loss function ──────────────────────────────────────────────────────────
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['bce', 'focal', 'margin', 'auto'],
                        help="Loss function. 'auto' = margin per TransE, focal per gli altri. "
                             "'bce' = BCE + label smoothing (CompGCN paper). "
                             "'focal' = Focal Loss + adversarial weighting (dataset sbilanciati). "
                             "'margin' = Pairwise margin loss (TransE paper). Default: auto.")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="Label smoothing per la BCE loss (es. 0.1). Ignorato se --loss focal/margin.")
    # focal-specific
    parser.add_argument('--alpha', type=float, default=0.25,
                        help="Focal loss alpha. Ignorato se --loss bce/margin.")
    parser.add_argument('--gamma', type=float, default=3.0,
                        help="Focal loss gamma. Ignorato se --loss bce/margin.")
    parser.add_argument('--alpha_adv', type=float, default=2.0,
                        help="Adversarial weighting temperature. Ignorato se --loss bce/margin.")
    # margin-specific
    parser.add_argument('--margin', type=float, default=1.0,
                        help="Margin for pairwise margin loss (TransE). Ignorato se --loss bce/focal.")
    # ──────────────────────────────────────────────────────────────────────────

    parser.add_argument('--eval_filtered', action='store_true', default=True)
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()
    main(args)