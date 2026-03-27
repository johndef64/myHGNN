"""
Train and evaluate heterogeneous GNN models for NODE CLASSIFICATION.

Uses modular encoder/decoder architecture:
  - Encoders: CompGCN, R-GCN, R-GAT
  - Decoder: NodeClassifier (MLP)

Supports benchmark datasets: AIFB, MUTAG, BGS, AM (from R-GCN paper).

# per caricare un custom TSV dataset, usare --dataset path/to/dataset.tsv (il loader riconosce automaticamente se è un dataset PyG o un TSV)
- tsv_path: path to triples TSV (head, interaction, tail)
- A companion file <tsv_path>.labels.tsv with (node_id, label) columns


Usage:
  python train_node_classification.py --dataset aifb --model rgcn --epochs 50
  python train_node_classification.py --dataset mutag --model rgcn --epochs 50 --runs 5
  python train_node_classification.py --dataset bgs --model rgcn --runs 10

  # GAT
  python train_node_classification.py --dataset aifb --model rgat --epochs 50
  python train_node_classification.py --dataset aifb --model rgat --epochs 200 --early_stopping --patience 30


  
  # custom dataset in TSV format
  python train_node_classification.py --dataset dataset/PathogenKG_n31.tsv --model rgcn --epochs 50

  python train_node_classification.py --dataset dataset/PathogenKG_n31.tsv --model rgcn --epochs 50 --config_name node-classification-128
  
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

from src.encoders import get_encoder
from src.decoders.node_classifier import NodeClassifier
from src.datasets.node_cls_datasets import load_node_cls_dataset, NODE_CLS_DATASETS
from src.utils import set_seed, graph_to_undirect, add_self_loops

_KGE_MODELS = {'distmult_kge', 'transe', 'rotate', 'node2vec'}

warnings.simplefilter(action='ignore')

BASE_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(model_name, dataset_info, config_name, models_params_path):
    with open(models_params_path, 'r') as f:
        models_params = json.load(f)

    if config_name not in models_params:
        print(f"[!] Config '{config_name}' not found, using 'default'")
        config_name = 'default'

    if model_name not in models_params[config_name]:
        if model_name in models_params.get('default', {}):
            print(f"[!] Model '{model_name}' not in config '{config_name}', using 'default' params")
            mp = models_params['default'][model_name]
        else:
            raise KeyError(f"Model '{model_name}' not found in config '{config_name}' or 'default'")
    else:
        mp = models_params[config_name][model_name]

    print(f"[i] Model params ({config_name}): {mp}")

    conv_hidden = {f'layer_{x}': mp[f'layer_{x}'] for x in range(mp['conv_layer_num'])}
    # KGE models have conv_layer_num=0 — out_dim comes from the embedding size directly
    out_dim = mp['mlp_out_layer'] if mp['conv_layer_num'] == 0 else conv_hidden[f"layer_{mp['conv_layer_num']-1}"]

    EncoderClass = get_encoder(model_name)

    num_relations = dataset_info['num_relations']

    if model_name == 'compgcn':
        encoder = EncoderClass(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=num_relations,
            conv_num_layers=mp['conv_layer_num'],
            opn=mp.get('opn', 'sub'),
            dropout=mp.get('dropout', 0.1),
            device=device,
        )
    elif model_name in _KGE_MODELS:
        encoder = EncoderClass(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=num_relations,
            conv_num_layers=mp['conv_layer_num'],
            device=device,
            dropout=mp.get('dropout', 0.0),
            walk_length=mp.get('walk_length', 20),
            context_size=mp.get('context_size', 10),
            walks_per_node=mp.get('walks_per_node', 10),
            p=mp.get('p', 1.0),
            q=mp.get('q', 1.0),
            pretrain_epochs=mp.get('pretrain_epochs', 5),
        )
    else:  # rgcn, rgat
        encoder_kwargs_init = dict(
            in_channels_dict=dataset_info['in_channels_dict'],
            mlp_hidden_channels_dict=mp['mlp_out_layer'],
            mlp_out_emb_size=mp['mlp_out_layer'],
            conv_hidden_channels=conv_hidden,
            num_nodes_per_type=dataset_info['num_nodes_per_type'],
            num_entities=dataset_info['num_entities'],
            num_relations=num_relations + 1,
            conv_num_layers=mp['conv_layer_num'],
            num_bases=mp['num_bases'],
            activation_function=F.relu,
            device=device,
        )
        if model_name == 'rgat':
            encoder_kwargs_init['dropout'] = mp.get('dropout', 0.1)
        encoder = EncoderClass(**encoder_kwargs_init)

    # RotatE entity embeddings are [N, 2*d] (real + imag concatenated)
    classifier_dim = 2 * out_dim if model_name == 'rotate' else out_dim

    decoder = NodeClassifier(
        hidden_dim=classifier_dim,
        num_classes=dataset_info['num_classes'],
        dropout=mp.get('dropout', 0.3),
        num_layers=1,  # single linear layer, as in R-GCN paper
    )

    return encoder, decoder, mp, out_dim


# ─── Training step ───────────────────────────────────────────────────────────

def train_step(encoder, decoder, optimizer, grad_norm,
               x_dict, edge_index, train_idx, train_y, **encoder_kwargs):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    logits = decoder(node_emb, rel_emb, train_idx)
    loss = decoder.compute_loss(logits, train_y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()), grad_norm
    )
    optimizer.step()

    preds = logits.argmax(dim=1).cpu().numpy()
    labels = train_y.cpu().numpy()
    acc = accuracy_score(labels, preds)

    return {'Loss': loss.item(), 'Accuracy': acc}


# ─── Evaluation step ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_step(encoder, decoder, x_dict, edge_index,
              node_idx, node_y, **encoder_kwargs):
    encoder.eval()
    decoder.eval()

    node_emb, rel_emb = encoder(x_dict, edge_index, **encoder_kwargs)
    logits = decoder(node_emb, rel_emb, node_idx)
    loss = decoder.compute_loss(logits, node_y)

    preds = logits.argmax(dim=1).cpu().numpy()
    labels = node_y.cpu().numpy()

    return {
        'Loss': loss.item(),
        'Accuracy': accuracy_score(labels, preds),
        'Macro-F1': f1_score(labels, preds, average='macro', zero_division=0),
        'Micro-F1': f1_score(labels, preds, average='micro', zero_division=0),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    print(f'[i] Device: {device}')
    print(f'[i] Dataset: {args.dataset}')
    print(f'[i] Model: {args.model}')

    models_params_path = './src/models_params.json'
    all_run_metrics = []

    for run_i in range(args.runs):
        seed = BASE_SEED + run_i
        set_seed(seed)
        print(f'\n{"="*60}\n[i] Run {run_i}/{args.runs} | seed={seed}')

        # Load dataset
        ds = load_node_cls_dataset(args.dataset, root=args.data_root, seed=seed)

        print(f"[i] Entities: {ds['num_entities']}, Relations: {ds['num_relations']}, "
              f"Classes: {ds['num_classes']}")
        print(f"[i] Train: {len(ds['train_idx'])}, Val: {len(ds['val_idx'])}, "
              f"Test: {len(ds['test_idx'])}")

        # Prepare edge triples for message passing
        edge_triples = ds['edge_triples']
        num_entities = ds['num_entities']
        num_relations = ds['num_relations']

        # For PyG Entities datasets (AIFB, MUTAG, BGS, AM), the edge types
        # already include both directions as separate relation types.
        # CompGCN also handles bidirectionality internally (w_in / w_out).
        # So we do NOT call graph_to_undirect here to avoid redundancy.
        # For RGCN/RGAT (which don't handle bidirectionality internally),
        # we still add reverse edges.
        triples_np = edge_triples.numpy()
        # KGE models ignore edge_index entirely; GNN models need directed/undirected edges.
        if args.model in ('rgcn', 'rgat'):
            triples_np = graph_to_undirect(triples_np, num_relations)
            triples_np = add_self_loops(triples_np, num_entities, num_relations)
        edge_index = torch.tensor(triples_np, dtype=torch.long).to(device)

        # Update num_relations based on actual edge types present
        actual_num_relations = int(edge_index[:, 1].max().item()) + 1
        ds['num_relations'] = actual_num_relations

        # Build model
        encoder, decoder, mp, _out_dim = build_model(
            args.model, ds, args.config_name, models_params_path
        )
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Prepare features
        features = {nt: (f.to(device) if f is not None else None)
                    for nt, f in ds['flattened_features'].items()}

        train_idx = ds['train_idx'].to(device)
        train_y = ds['train_y'].to(device)
        val_idx = ds['val_idx'].to(device)
        val_y = ds['val_y'].to(device)
        test_idx = ds['test_idx'].to(device)
        test_y = ds['test_y'].to(device)

        # Encoder kwargs (RGAT needs edges sorted by relation type for change_points)
        encoder_kwargs = {}
        if args.model == 'rgat':
            # Sort edges by relation type — required by HRGATConv
            sort_idx = edge_index[:, 1].argsort()
            edge_index = edge_index[sort_idx]

            rel_ids = edge_index[:, 1]
            encoder_kwargs['change_points'] = torch.cat([
                torch.tensor([0], device=device),
                (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
                torch.tensor([rel_ids.size(0)], device=device)
            ])

        # Optimizer
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(all_params, lr=mp['learning_rate'],
                                     weight_decay=mp.get('weight_decay', 0.0))

        # Training loop
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        save_dir = None
        if not args.dry_run:
            ts = time.strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join('models', f'nc_{args.dataset}_{args.model}_{ts}')
            os.makedirs(save_dir, exist_ok=True)

        with trange(1, args.epochs + 1, desc=f'Run {run_i}') as pbar:
            for epoch in pbar:
                train_m = train_step(
                    encoder, decoder, optimizer, mp.get('grad_norm', 1.0),
                    features, edge_index, train_idx, train_y, **encoder_kwargs
                )

                val_m = {'Accuracy': 0, 'Loss': 0}
                if epoch % args.evaluate_every == 0:
                    val_m = eval_step(
                        encoder, decoder, features, edge_index,
                        val_idx, val_y, **encoder_kwargs
                    )

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

        # Load best model and test
        if best_state is not None:
            encoder.load_state_dict({k: v.to(device) for k, v in best_state['encoder'].items()})
            decoder.load_state_dict({k: v.to(device) for k, v in best_state['decoder'].items()})

        test_m = eval_step(
            encoder, decoder, features, edge_index,
            test_idx, test_y, **encoder_kwargs
        )

        print(f"Run {run_i} | Test Accuracy: {test_m['Accuracy']:.4f}, "
              f"Macro-F1: {test_m['Macro-F1']:.4f}, Micro-F1: {test_m['Micro-F1']:.4f}")

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
    parser = argparse.ArgumentParser(description='Node Classification on Heterogeneous Graphs')
    parser.add_argument('--dataset', type=str, default='aifb',
                        help=f'Dataset name: {NODE_CLS_DATASETS} or path to TSV')
    parser.add_argument('--data_root', type=str, default='dataset/')
    parser.add_argument('-m', '--model', type=str, default='compgcn',
                        choices=['rgcn', 'rgat', 'compgcn',
                                 'distmult_kge', 'transe', 'rotate', 'node2vec'])
    parser.add_argument('--config_name', type=str, default='node-classification-32')
    parser.add_argument('-r', '--runs', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-p', '--patience', type=int, default=30)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--evaluate_every', type=int, default=1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()
    main(args)


