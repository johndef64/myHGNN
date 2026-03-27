import requests
import json
import os

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

# Scarica il file GO in formato JSON
url = "http://purl.obolibrary.org/obo/go.json"
response = requests.get(url)

if response.status_code == 200:
    go_data = response.json()
    print("File GO JSON scaricato con successo")
else:
    print(f"Errore nel download: {response.status_code}")


#%%

def extract_go_relationships(go_data):
    """
    Estrae tutte le relazioni GO:GO dal file JSON
    """
    relationships = []
    
    # Naviga nella struttura JSON
    if 'graphs' in go_data:
        for graph in go_data['graphs']:
            if 'nodes' in graph:
                for node in graph['nodes']:
                    if 'id' in node and node['id'].startswith('http://purl.obolibrary.org/obo/GO_'):
                        go_id = node['id'].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')
                        go_name = node.get('lbl', 'Unknown')
                        
                        # Estrai relazioni is_a (subClassOf)
                        if 'meta' in node and 'subsets' in node['meta']:
                            for subset in node['meta']['subsets']:
                                relationships.append({
                                    'child': go_id,
                                    'parent': subset,
                                    'relationship_type': 'subset',
                                    'child_name': go_name
                                })
            
            # Estrai edges (relazioni)
            if 'edges' in graph:
                for edge in graph['edges']:
                    if ('sub' in edge and 'obj' in edge and 
                        edge['sub'].startswith('http://purl.obolibrary.org/obo/GO_') and
                        edge['obj'].startswith('http://purl.obolibrary.org/obo/GO_')):
                        
                        child_id = edge['sub'].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')
                        parent_id = edge['obj'].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')
                        
                        # Determina il tipo di relazione
                        rel_type = 'is_a'  # default
                        if 'pred' in edge:
                            if 'part_of' in edge['pred']:
                                rel_type = 'part_of'
                            elif 'regulates' in edge['pred']:
                                rel_type = 'regulates'
                            elif 'negatively_regulates' in edge['pred']:
                                rel_type = 'negatively_regulates'
                            elif 'positively_regulates' in edge['pred']:
                                rel_type = 'positively_regulates'
                        
                        relationships.append({
                            'child': child_id,
                            'parent': parent_id,
                            'relationship_type': rel_type,
                            'child_name': '',
                            'parent_name': ''
                        })
    
    return relationships

# Estrai le relazioni
# all_relationships = extract_go_relationships(go_data)
# print(f"Numero totale di relazioni estratte: {len(all_relationships)}")

# %%
# all_relationships[:10]  # Mostra le prime 10 relazioni estratte


#%%

def extract_go_triples_improved(go_data):
    triples = []
    go_terms = {}
    
    # Prima passa: raccogli tutti i termini GO
    for item in go_data.get('graphs', [{}])[0].get('nodes', []):
        if 'id' in item and 'GO_' in item.get('id', ''):
            go_id = item['id'].split('/')[-1].replace('_', ':')
            go_terms[go_id] = item.get('lbl', '')
    
    # Seconda passa: estrai le relazioni
    for edge in go_data.get('graphs', [{}])[0].get('edges', []):
        if 'sub' in edge and 'obj' in edge:
            child_uri = edge['sub']
            parent_uri = edge['obj']
            
            # Converti URI in formato GO:XXXXXXX
            if 'GO_' in child_uri and 'GO_' in parent_uri:
                child_id = child_uri.split('/')[-1].replace('_', ':')
                parent_id = parent_uri.split('/')[-1].replace('_', ':')
                
                # Analizza il predicato più accuratamente
                pred = edge.get('pred', '')
                
                if 'rdfs:subClassOf' in pred or pred.endswith('subClassOf'):
                    rel_type = 'is_a'
                elif 'BFO_0000050' in pred or 'part_of' in pred:
                    rel_type = 'part_of'
                elif 'RO_0002211' in pred or 'regulates' in pred:
                    rel_type = 'regulates'
                elif 'RO_0002212' in pred:
                    rel_type = 'negatively_regulates'
                elif 'RO_0002213' in pred:
                    rel_type = 'positively_regulates'
                else:
                    # Stampa il predicato per debug
                    print(f"Predicato non riconosciuto: {pred}")
                    rel_type = 'is_a'  # Default per relazioni parent-child
                
                triples.append((child_id, rel_type, parent_id))
    
    return triples, go_terms

# Esecuzione
relations = []
if go_data:
    go_triples, terms = extract_go_triples_improved(go_data)
    
    print(f"Totale triple estratte: {len(go_triples)}")
    print("\nPrime 15 triple (child, relazione, parent):")
    for i, triple in enumerate(go_triples[:15]):
        child, rel, parent = triple
        child_name = terms.get(child, '')
        parent_name = terms.get(parent, '')
        print(f"{i+1}. {child} --{rel}--> {parent}")
        print(f"   {child_name} --{rel}--> {parent_name}")

        print()

# %%
# head    interaction	tail	source	type
# GO::GO:0000011	is_a	GO::GO:0048308	GO	GO-GO

import requests
import json
import os
import pandas as pd

# 1. Scarica il file GO JSON
url = "http://purl.obolibrary.org/obo/go.json"
response = requests.get(url)
go_data = response.json()

# 2. Estrai le relazioni GO:GO
def extract_go_relations(go_data):
    relations = []
    
    for edge in go_data.get('graphs', [{}])[0].get('edges', []):
        if 'sub' in edge and 'obj' in edge:
            child_uri = edge['sub']
            parent_uri = edge['obj']
            
            if 'GO_' in child_uri and 'GO_' in parent_uri:
                child_id = child_uri.split('/')[-1].replace('_', ':')
                parent_id = parent_uri.split('/')[-1].replace('_', ':')
                
                pred = edge.get('pred', '')
                if 'subClassOf' in pred:
                    rel_type = 'is_a'
                elif 'BFO_0000050' in pred:
                    rel_type = 'part_of'
                elif 'RO_0002211' in pred:
                    rel_type = 'regulates'
                elif 'RO_0002212' in pred:
                    rel_type = 'negatively_regulates'
                elif 'RO_0002213' in pred:
                    rel_type = 'positively_regulates'
                else:
                    rel_type = 'is_a'
                
                relations.append((child_id, rel_type, parent_id))
    
    return relations

# 3. Estrai le relazioni
go_relations = extract_go_relations(go_data)

# 4. Salva nel formato TSV richiesto
def save_to_tsv(relations):
    df_data = []
    
    for child_id, relationship_type, parent_id in relations:
        df_data.append({
            'head': f'GO::{child_id}',
            'interaction': relationship_type,
            'tail': f'GO::{parent_id}',
            'source': 'GO',
            'type': 'GO-GO'
        })
    
    df = pd.DataFrame(df_data)
    
    # Crea cartella dataset
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    # Salva file TSV
    output_file = 'dataset/go_relationships.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"✅ File salvato: {output_file}")
    print(f"📊 Relazioni salvate: {len(df):,}")
    
    return output_file

# Esegui il salvataggio
output_file = save_to_tsv(go_relations)

