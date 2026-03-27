# %%
# head    interaction	tail	source	type
# GO::GO:0000011	is_a	GO::GO:0048308	GO	GO-GO

import requests
import json
import os
import pandas as pd

# Change to parent directory
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
print("Current directory:", os.getcwd())


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


# %%
