# Generalization: Link Prediction → Node Classification → Graph Classification

This repository has been refactored from a link-prediction-only framework into a general-purpose
heterogeneous GNN template supporting **three task types**:

1. **Link Prediction** — predict missing edges in a knowledge graph (DistMult decoder)
2. **Node Classification** — classify nodes in a heterogeneous graph (MLP decoder)
3. **Graph Classification** — classify entire graphs (pooling + MLP decoder)

---

## Architecture Overview

```
                    ┌──────────────────┐
                    │   Encoder        │
                    │  (CompGCN/RGCN/  │
                    │   RGAT)          │
                    │                  │
                    │  x_dict + edges  │
                    │       ↓          │
                    │  (node_emb,      │
                    │   rel_emb)       │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
     ┌────────────┐  ┌─────────────┐  ┌──────────────┐
     │  DistMult  │  │    Node     │  │    Graph     │
     │  Decoder   │  │ Classifier  │  │ Classifier   │
     │            │  │             │  │              │
     │ (h,r,t)→s  │  │ emb[i]→cls  │  │ pool→cls     │
     └────────────┘  └─────────────┘  └──────────────┘
     Link Prediction  Node Classif.   Graph Classif.
```

### Key Design: Encoder returns `(node_embeddings, relation_embeddings)`

All encoders share the same interface:
```python
node_emb, rel_emb = encoder(x_dict, edge_index, **kwargs)
```

Decoders are task-specific and consume these embeddings differently.

---

## New Repository Structure

```
myGNN/
├── src/
│   ├── encoders/                    # GNN encoders (shared across tasks)
│   │   ├── __init__.py
│   │   ├── compgcn.py               # CompGCN encoder
│   │   ├── rgcn.py                  # R-GCN encoder
│   │   └── rgat.py                  # R-GAT encoder
│   │
│   ├── decoders/                    # Task-specific decoders
│   │   ├── __init__.py
│   │   ├── distmult.py              # DistMult for link prediction
│   │   ├── node_classifier.py       # MLP for node classification
│   │   └── graph_classifier.py      # Pooling + MLP for graph classification
│   │
│   ├── datasets/                    # Dataset loaders
│   │   ├── __init__.py
│   │   ├── kg_dataset.py            # TSV triple loader (link prediction)
│   │   ├── node_cls_datasets.py     # AIFB/MUTAG/BGS/AM (node classification)
│   │   └── graph_cls_datasets.py    # TUDataset molecular graphs (graph classification)
│   │
│   ├── utils.py                     # Shared utilities (unchanged)
│   ├── evaluation_metrics_filtered.py
│   └── models_params.json
│
├── train_link_prediction.py         # Training script for link prediction
├── train_node_classification.py     # Training script for node classification
├── train_graph_classification.py    # Training script for graph classification
│
├── train_and_eval.py                # Original script (kept for backward compat)
├── dataset/                         # Data directory
└── ...
```

---

## Quick Start

### Link Prediction (on DRKG or PathogenKG)
```bash
python train_link_prediction.py --model compgcn --epochs 400 --task TARGET
python train_link_prediction.py --model rgcn --tsv dataset/drkg/drkg_reduced.tsv --task Compound-Gene
```

### Node Classification (on AIFB/MUTAG/BGS/AM benchmarks)
```bash
python train_node_classification.py --dataset aifb --model compgcn --epochs 50
python train_node_classification.py --dataset mutag --model rgcn --runs 10 --epochs 50
python train_node_classification.py --dataset bgs --model compgcn --runs 5
python train_node_classification.py --dataset am --model compgcn --runs 5
```

### Graph Classification (on molecular benchmarks)
```bash
python train_graph_classification.py --dataset mutag --epochs 200 --runs 10
python train_graph_classification.py --dataset ptc_mr --epochs 200 --pooling mean_max
python train_graph_classification.py --dataset proteins --epochs 100 --hidden_dim 128
```

---

## Benchmark Datasets

### Link Prediction
| Dataset | Entities | Triples | Relations | Format |
|---------|----------|---------|-----------|--------|
| PathogenKG (n31) | ~180K | ~3M | 13 | TSV |
| DRKG | ~97K | ~5.9M | 107 | TSV |

### Node Classification (R-GCN benchmarks)
| Dataset | Entities | Triples | Relations | Classes | Task |
|---------|----------|---------|-----------|---------|------|
| AIFB | ~8K | ~29K | 45 | 4 | Researcher affiliation |
| MUTAG | ~23K | ~74K | 23 | 2 | Molecule mutagenicity |
| BGS | ~333K | ~916K | 103 | 2 | Rock lithogenesis |
| AM | ~1.6M | ~5.9M | 133 | 11 | Artifact category |

### Graph Classification (molecular benchmarks)
| Dataset | Graphs | Avg Nodes | Edge Types | Classes | Task |
|---------|--------|-----------|------------|---------|------|
| MUTAG | 188 | ~18 | 4 | 2 | Mutagenicity |
| PTC_MR | 344 | ~26 | 4 | 2 | Carcinogenicity |
| PROTEINS | 1113 | ~39 | 1 | 2 | Enzyme vs non-enzyme |
| NCI1 | 4110 | ~30 | 1 | 2 | Anti-cancer activity |

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Link Prediction | AUROC, AUPRC, MRR, Hits@1, Hits@3, Hits@10 |
| Node Classification | Accuracy, Macro-F1, Micro-F1 |
| Graph Classification | Accuracy, Macro-F1, Micro-F1 |
