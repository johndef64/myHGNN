#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.models import GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score


class KnowledgeGraphEncoder(nn.Module):
    """
    Graph encoder for knowledge graph node representation learning
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super(KnowledgeGraphEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        return x

class VariationalKGEncoder(nn.Module):
    """
    Variational encoder for knowledge graph autoencoder
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super(VariationalKGEncoder, self).__init__()
        
        self.shared_encoder = KnowledgeGraphEncoder(
            in_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
        
        # Mean and log variance layers
        self.mu_layer = GCNConv(hidden_channels, out_channels)
        self.logvar_layer = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Shared encoding
        h = self.shared_encoder(x, edge_index)
        
        # Compute mean and log variance
        mu = self.mu_layer(h, edge_index)
        logvar = self.logvar_layer(h, edge_index)
        
        return mu, logvar

class KnowledgeGraphAutoEncoder:
    """
    Complete Knowledge Graph AutoEncoder implementation
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 variational=False, num_layers=2, dropout=0.1):
        self.variational = variational
        
        if variational:
            encoder = VariationalKGEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
            self.model = VGAE(encoder)
        else:
            encoder = KnowledgeGraphEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
            self.model = GAE(encoder)
    
    def get_model(self):
        return self.model

class KGAutoEncoderTrainer:
    """
    Training class for Knowledge Graph AutoEncoder
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Encode nodes
        z = self.model.encode(data.x, data.edge_index)
        
        # Compute reconstruction loss
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # Reconstruction loss
        recon_loss = self.model.recon_loss(z, pos_edge_index, neg_edge_index)
        
        # KL divergence loss (for variational autoencoder)
        if hasattr(self.model, 'kl_loss'):
            kl_loss = self.model.kl_loss() / data.num_nodes
            loss = recon_loss + kl_loss
        else:
            loss = recon_loss
            
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            z = self.model.encode(data.x, data.edge_index)
            
            # Generate negative samples for evaluation
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
            
            # Get predictions
            pos_pred = self.model.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.model.decoder(z, neg_edge_index, sigmoid=True)
            
            # Compute metrics
            pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
            y_true = torch.cat([
                torch.ones(pos_pred.size(0)),
                torch.zeros(neg_pred.size(0))
            ], dim=0).cpu().numpy()
            
            auc = roc_auc_score(y_true, pred)
            ap = average_precision_score(y_true, pred)
            
            return auc, ap
    
    def get_node_embeddings(self, data):
        """Extract node embeddings from trained model"""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            z = self.model.encode(data.x, data.edge_index)
            return z.cpu().numpy()

# Example usage and training
def create_sample_knowledge_graph():
    """Create a sample knowledge graph for demonstration"""
    # Sample knowledge graph with 100 nodes and random features
    num_nodes = 100
    num_features = 16
    
    # Random node features
    x = torch.randn(num_nodes, num_features)
    
    # Create edges (knowledge graph triples)
    # In a real KG, these would be subject-predicate-object relationships
    edge_index = torch.randint(0, num_nodes, (2, 300))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return Data(x=x, edge_index=edge_index)


def load_knowledge_graph_from_file(file_path):
    """Load a knowledge graph from a file in PyTorch Geometric format"""
    from torch_geometric.data import Data
    import torch
    import pandas as pd
    
    # Load data from TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Create node features (dummy features for demonstration)
    num_nodes = df['head'].nunique() + df['tail'].nunique()
    x = torch.randn(num_nodes, 16)  # Random features
    
    # Create edge index
    head_indices = df['head'].apply(lambda x: int(x.split('::')[-1])).values
    tail_indices = df['tail'].apply(lambda x: int(x.split('::')[-1])).values
    edge_index = torch.tensor([head_indices, tail_indices], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


def load_knowledge_graph_with_mapping(file_path):
    """
    Carica il knowledge graph creando una mappatura tra entità e indici numerici
    """
    df = pd.read_csv(file_path, sep='\t')

    # Estrai tutte le entità uniche
    all_entities = set()
    all_entities.update(df['head'].unique())
    all_entities.update(df['tail'].unique())
    
    # Crea mappatura entità -> indice
    entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}
    
    # Converti le entità in indici
    head_indices = df['head'].map(entity_to_idx).values
    tail_indices = df['tail'].map(entity_to_idx).values
    
    # Crea l'edge_index
    edge_index = torch.tensor([head_indices, tail_indices], dtype=torch.long)

    # Create node features (dummy features for demonstration)
    num_nodes = df['head'].nunique() + df['tail'].nunique()
    x = torch.randn(num_nodes, 16)  # Random features
    
    #return edge_index, entity_to_idx, idx_to_entity
    return Data(x=x, edge_index=edge_index), entity_to_idx, idx_to_entity



def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample knowledge graph
    data = create_sample_knowledge_graph()

    file_path = r'G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\pathogenkg\PathogenKG_83332.tsv'
    file_path = r'G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\pathogenkg\VitaGraph_human_extended.tsv'
    data, entity_to_idx, idx_to_entity = load_knowledge_graph_with_mapping(file_path)
    print(f"Knowledge Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Initialize autoencoder
    kg_autoencoder = KnowledgeGraphAutoEncoder(
        in_channels=data.num_features,
        hidden_channels=32,
        out_channels=16,
        variational=True,  # Use variational autoencoder
        num_layers=3,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = KGAutoEncoderTrainer(kg_autoencoder.get_model(), device)
    
    # Training loop
    print("Training Knowledge Graph AutoEncoder...")
    for epoch in range(200):
        loss = trainer.train_epoch(data)
        
        if epoch % 50 == 0:
            auc, ap = trainer.evaluate(data)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
    
    # Extract final node embeddings
    node_embeddings = trainer.get_node_embeddings(data)
    print(f"Node embeddings shape: {node_embeddings.shape}")
    
    return trainer, node_embeddings, entity_to_idx, idx_to_entity

if __name__ == "__main__":
    trainer, embeddings, entity_to_idx, idx_to_entity  = main()

#%%
embeddings[:5]  # Show first 5 node embeddings
from tqdm import tqdm


print(f"Entity to Index Mapping: {list(entity_to_idx.items())[:5]}")  # Show first 5 mappings
print(f"Index to Entity Mapping: {list(idx_to_entity.items())[:5]}")  # Show first 5 mappings
print(f"Total entities: {len(entity_to_idx)}")  # Total number of entities in the knowledge graph
print(f"Total nodes: {embeddings.shape[0]}")  # Total number of nodes in the knowledge graph

def calcualte_similarity(embeddings, entity_to_idx, idx_to_entity):
    """
    Calculate cosine similarity between node embeddings
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Create a mapping of entity pairs to their similarity scores
    similarity_scores = defaultdict(dict)
    
    for i in tqdm(range(len(embeddings))):
        for j in range(len(embeddings)):
            if i != j and j < len(idx_to_entity) and i < len(idx_to_entity) :
                entity_i = idx_to_entity[i]
                entity_j = idx_to_entity[j]
                similarity_scores[entity_i][entity_j] = sim_matrix[i][j]
    
    return similarity_scores

similarity_scores = calcualte_similarity(embeddings, entity_to_idx, idx_to_entity)

#%%
# get top 5 most similar entities for a given entity
def get_top_similar_entities(entity, similarity_scores, top_n=5):
    """
    Get top N most similar entities for a given entity
    """
    if entity not in similarity_scores:
        return []
    
    # Sort entities by similarity score
    sorted_entities = sorted(similarity_scores[entity].items(), key=lambda x: x[1], reverse=True)
    
    # Return top N entities
    return sorted_entities[:top_n]

entity = "Compound::Pubchem:10041129"
entity = "Gene::NCBI:5141"  # Example entity
top_similar_entities = get_top_similar_entities(entity, similarity_scores, top_n=5)
print(f"Top 5 similar entities for {entity}: {top_similar_entities}")


# %%
entity_to_idx[entity]  # Get index of the entity
# %%
len(idx_to_entity)
idx_to_entity[5]  # Show first 5 entities in the index to entity mapping
# %%
len(embeddings),  len(idx_to_entity), len(entity_to_idx)  # Check if embeddings match the number of entities

#%%
# Save the model and embeddings
import torch
import json
def save_model_and_embeddings(model, embeddings, entity_to_idx, idx_to_entity, model_path='kg_autoencoder.pth', embeddings_path='embeddings.pt'):
    """
    Save the trained model and embeddings to files
    """
    torch.save(model.state_dict(), model_path)
    torch.save(embeddings, embeddings_path)
    
    # Save entity mappings
    with open('entity_to_idx.json', 'w') as f:
        json.dump(entity_to_idx, f)
    
    with open('idx_to_entity.json', 'w') as f:
        json.dump(idx_to_entity, f)

#%%
import pickle
import torch

def save_with_pickle(trainer, embeddings, entity_to_idx, idx_to_entity, 
                    filename='kg_autoencoder_complete.pkl'):
    """
    Salva tutto in un singolo file pickle
    """
    # Prepara il dizionario con tutti i dati
    save_data = {
        'model_state_dict': trainer.model.state_dict(),
        'embeddings': embeddings,
        'entity_to_idx': entity_to_idx,
        'idx_to_entity': idx_to_entity,
        'model_type': type(trainer.model).__name__,
        'num_entities': len(entity_to_idx),
        'embedding_dim': embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0])
    }
    
    # Salva tutto in un file
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Tutto salvato in: {filename}")

def load_with_pickle(filename='kg_autoencoder_complete.pkl'):
    """
    Carica tutto dal file pickle
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data

# Utilizzo
save_with_pickle(trainer, embeddings, 
                entity_to_idx, 
                idx_to_entity)

# Per caricare
loaded_data = load_with_pickle()
print(f"Caricati {loaded_data['num_entities']} entità")
print(f"Embeddings shape: {loaded_data['embeddings'].shape}")

#%%