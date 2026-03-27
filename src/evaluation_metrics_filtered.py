"""
Evaluation metrics for Knowledge Graph Link Prediction — Filtered Setting.

Contains two variants:
  - evaluation_metrics_filtered_fullgraph: ranking against ALL nodes in the graph (strict)
- evaluation_metrics_filtered_typeconstrained: ranking only against nodes
of the same type (head vs head pool, tail vs tail pool) — as done
in the papers on Hetionet and GP-KG for drug repurposing.
"""

import torch
from collections import defaultdict


def build_positive_maps(all_target_triplets):
    """
    Builds maps of known positives for the filtered setting.
    
    Args:
        all_target_triplets: tensor (N, 3) con TUTTE le triple positive 
                             (train + val + test) della relazione target.
    
    Returns:
        all_positives_tail: dict (h, r) -> set of t
        all_positives_head: dict (r, t) -> set of h
    """
    all_positives_tail = defaultdict(set)
    all_positives_head = defaultdict(set)
    
    for i in range(all_target_triplets.size(0)):
        h = all_target_triplets[i, 0].item()
        r = all_target_triplets[i, 1].item()
        t = all_target_triplets[i, 2].item()
        all_positives_tail[(h, r)].add(t)
        all_positives_head[(r, t)].add(h)
    
    return all_positives_tail, all_positives_head


def _compute_ranks(model, embeddings, test_triplets, candidate_nodes, 
                   all_positives_map, node_to_idx, device, mode, verbose):
    """
    Internal helper: calculates ranks for head or tail prediction.

    Args:
    mode: 'tail' or 'head'
    candidate_nodes: tensor of candidate nodes for ranking
    all_positives_map: dict for filtering (tail_map or head_map)
    """
    ranks = []
    skipped = 0
    num_candidates = candidate_nodes.size(0)
    
    # Mappa candidato -> indice nel vettore candidate_nodes
    cand_to_idx = {}
    for i in range(num_candidates):
        cand_to_idx[candidate_nodes[i].item()] = i
    
    with torch.no_grad():
        for i in range(test_triplets.size(0)):
            h, r, t = test_triplets[i]
            h_i, r_i, t_i = h.item(), r.item(), t.item()
            
            if mode == 'tail':
                true_node = t_i
                lookup_key = (h_i, r_i)
                # Costruisci triple: (h, r, candidate) per ogni candidato
                candidates = torch.stack([
                    h.expand(num_candidates).to(device),
                    r.expand(num_candidates).to(device),
                    candidate_nodes
                ], dim=1)
            else:  # head
                true_node = h_i
                lookup_key = (r_i, t_i)
                # Costruisci triple: (candidate, r, t) per ogni candidato
                candidates = torch.stack([
                    candidate_nodes,
                    r.expand(num_candidates).to(device),
                    t.expand(num_candidates).to(device)
                ], dim=1)
            
            # Il nodo vero deve essere tra i candidati
            if true_node not in cand_to_idx:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] Tripla {i}: nodo {true_node} non nel pool candidati ({mode})")
                continue
            
            scores = model.distmult(embeddings, candidates)
            true_score = scores[cand_to_idx[true_node]]
            
            # Filtered: maschera i positivi noti tranne quello corrente
            filter_mask = torch.ones(num_candidates, dtype=torch.bool, device=device)
            for known_node in all_positives_map.get(lookup_key, set()):
                if known_node != true_node and known_node in cand_to_idx:
                    filter_mask[cand_to_idx[known_node]] = False
            
            filtered_scores = scores[filter_mask]
            rank = (filtered_scores >= true_score).sum().item()
            rank = max(rank, 1)
            ranks.append(rank)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  [{mode}] Valutate {i+1}/{test_triplets.size(0)} (rank={rank})")
    
    return ranks, skipped


def _aggregate_results(tail_ranks, head_ranks, skipped, hits_k, device):
    """Helper: aggrega i rank in metriche MRR e Hits@K."""
    if len(tail_ranks) == 0 and len(head_ranks) == 0:
        print("  [ERROR] Nessuna tripla valutata!")
        return {
            'mrr': 0.0, 'mrr_tail': 0.0, 'mrr_head': 0.0,
            **{f'hits@{k}': 0.0 for k in hits_k},
            **{f'hits@{k}_tail': 0.0 for k in hits_k},
            **{f'hits@{k}_head': 0.0 for k in hits_k},
            'num_evaluated': 0, 'num_skipped': skipped
        }
    
    tail_ranks_t = torch.tensor(tail_ranks, dtype=torch.float, device=device) if tail_ranks else torch.tensor([], dtype=torch.float, device=device)
    head_ranks_t = torch.tensor(head_ranks, dtype=torch.float, device=device) if head_ranks else torch.tensor([], dtype=torch.float, device=device)
    all_ranks_t = torch.cat([tail_ranks_t, head_ranks_t])
    
    results = {
        'mrr': (1.0 / all_ranks_t).mean().item() if all_ranks_t.numel() > 0 else 0.0,
        'mrr_tail': (1.0 / tail_ranks_t).mean().item() if tail_ranks_t.numel() > 0 else 0.0,
        'mrr_head': (1.0 / head_ranks_t).mean().item() if head_ranks_t.numel() > 0 else 0.0,
        'num_evaluated': max(len(tail_ranks), len(head_ranks)),
        'num_skipped': skipped,
    }
    
    for k in hits_k:
        results[f'hits@{k}'] = (all_ranks_t <= k).float().mean().item() if all_ranks_t.numel() > 0 else 0.0
        results[f'hits@{k}_tail'] = (tail_ranks_t <= k).float().mean().item() if tail_ranks_t.numel() > 0 else 0.0
        results[f'hits@{k}_head'] = (head_ranks_t <= k).float().mean().item() if head_ranks_t.numel() > 0 else 0.0
    
    return results


# ============================================================
# VERSIONE 1: Full-graph filtered (strict, ranking vs ALL nodes)
# ============================================================

def evaluation_metrics_filtered_fullgraph(
    model, 
    embeddings, 
    all_target_triplets,
    test_triplets,
    all_graph_nodes,
    device, 
    hits_k=[1, 3, 10],
    verbose=False
):
    """
    MRR and Hits@K with filtered setting — ranking against ALL nodes in the graph.

    This is the most rigorous version: for each triple (h, r, t), the true tail
    competes against all ~47K nodes in the graph, not just those of the same type.
    It produces lower but more conservative metrics.
    """
    model.eval()
    candidate_nodes = all_graph_nodes.to(device)
    
    node_to_idx = {candidate_nodes[i].item(): i for i in range(candidate_nodes.size(0))}
    all_positives_tail, all_positives_head = build_positive_maps(all_target_triplets)
    
    tail_ranks, skip_t = _compute_ranks(
        model, embeddings, test_triplets, candidate_nodes,
        all_positives_tail, node_to_idx, device, 'tail', verbose
    )
    head_ranks, skip_h = _compute_ranks(
        model, embeddings, test_triplets, candidate_nodes,
        all_positives_head, node_to_idx, device, 'head', verbose
    )
    
    total_skipped = skip_t + skip_h
    results = _aggregate_results(tail_ranks, head_ranks, total_skipped, hits_k, device)
    results['eval_mode'] = 'full_graph'
    results['num_tail_candidates'] = candidate_nodes.size(0)
    results['num_head_candidates'] = candidate_nodes.size(0)
    return results


# ============================================================
# VERSIONE 2: Type-constrained filtered (relaxed, come Hetionet)
# ============================================================

def evaluation_metrics_filtered_typeconstrained(
    model, 
    embeddings, 
    all_target_triplets,
    test_triplets,
    all_graph_nodes,
    device, 
    hits_k=[1, 3, 10],
    verbose=False
):
    """
    MRR and Hits@K with filtered setting — TYPE-CONSTRAINED ranking.

    Instead of ranking against all nodes in the graph:
      - Tail prediction: ranks only against nodes that appear as TAIL
        in the target relation (e.g., only Disease for TREATMENT)
      - Head prediction: ranks only against nodes that appear as HEAD
        nella relazione target (es. solo Compound per TREATMENT)
    
    Pools are automatically extracted from all_target_triplets.
    This is the setting used by the papers on Hetionet and GP-KG.
    
    SAME SIGNATURE as evaluation_metrics_filtered_fullgraph → drop-in replacement.
    The all_graph_nodes parameter is ignored; pools are derived
    from all_target_triplets.
    """
    model.eval()
    
    # Estrai i pool di candidati dai ruoli nella relazione target
    # Head pool = tutti i nodi che appaiono come head in all_target_triplets
    # Tail pool = tutti i nodi che appaiono come tail in all_target_triplets
    all_heads = all_target_triplets[:, 0]
    all_tails = all_target_triplets[:, 2]
    
    head_pool = torch.unique(all_heads).to(device)  # es. Compound
    tail_pool = torch.unique(all_tails).to(device)  # es. Disease / ExtGene
    
    if verbose:
        print(f"  [TYPE-CONSTRAINED] Head pool: {head_pool.size(0)} nodi, "
              f"Tail pool: {tail_pool.size(0)} nodi")
    
    all_positives_tail, all_positives_head = build_positive_maps(all_target_triplets)
    
    # Tail prediction: ranka t contro il tail_pool (es. solo Disease)
    tail_node_to_idx = {tail_pool[i].item(): i for i in range(tail_pool.size(0))}
    tail_ranks, skip_t = _compute_ranks(
        model, embeddings, test_triplets, tail_pool,
        all_positives_tail, tail_node_to_idx, device, 'tail', verbose
    )
    
    # Head prediction: ranka h contro il head_pool (es. solo Compound)
    head_node_to_idx = {head_pool[i].item(): i for i in range(head_pool.size(0))}
    head_ranks, skip_h = _compute_ranks(
        model, embeddings, test_triplets, head_pool,
        all_positives_head, head_node_to_idx, device, 'head', verbose
    )
    
    total_skipped = skip_t + skip_h
    results = _aggregate_results(tail_ranks, head_ranks, total_skipped, hits_k, device)
    results['eval_mode'] = 'type_constrained'
    results['num_tail_candidates'] = tail_pool.size(0)
    results['num_head_candidates'] = head_pool.size(0)
    return results



# Alias per compatibilità con il codice esistente
evaluation_metrics_filtered_relaxed = evaluation_metrics_filtered_typeconstrained


# ============================================
# Helper per stampare i risultati
# ============================================
def print_metrics(results, title="Evaluation Results"):
    """Stampa le metriche in formato leggibile."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    mode = results.get('eval_mode', 'unknown')
    n_tail = results.get('num_tail_candidates', '?')
    n_head = results.get('num_head_candidates', '?')
    print(f"  Mode: {mode} | Tail pool: {n_tail} | Head pool: {n_head}")
    print(f"  Triple valutate: {results['num_evaluated']}"
          f" (saltate: {results['num_skipped']})")
    print(f"  MRR (overall):   {results['mrr']:.4f}")
    print(f"  MRR (tail):      {results['mrr_tail']:.4f}")
    print(f"  MRR (head):      {results['mrr_head']:.4f}")
    for key in sorted(results.keys()):
        if key.startswith('hits@') and '_' not in key:
            k = key.replace('hits@', '')
            print(f"  Hits@{k} (overall): {results[key]:.4f}")
            print(f"  Hits@{k} (tail):    {results.get(f'hits@{k}_tail', 0):.4f}")
            print(f"  Hits@{k} (head):    {results.get(f'hits@{k}_head', 0):.4f}")
    print(f"{'='*60}\n")


# ============================================
# Esempio di utilizzo
# ============================================
"""
# 1. Collect ALL triples from the target relation (train + val + test)
all_target_triplets = torch.cat([
    train_target_triplets,
    val_target_triplets,
    test_target_triplets
], dim=0)

# 2. All nodes in the graph (for the full-graph version)
all_graph_nodes = torch.arange(num_entities)

# 3. Evaluate both versions
results_strict = evaluation_metrics_filtered_fullgraph(
    model, embeddings, all_target_triplets, test_positives,
    all_graph_nodes, device, verbose=True
)
print_metrics(results_strict, "DRKG TREATMENT — Full Graph Filtered")

results_tc = evaluation_metrics_filtered_typeconstrained(
    model, embeddings, all_target_triplets, test_positives,
    all_graph_nodes, device, verbose=True  # all_graph_nodes viene ignorato
)
print_metrics(results_tc, "DRKG TREATMENT — Type-Constrained Filtered")
"""

