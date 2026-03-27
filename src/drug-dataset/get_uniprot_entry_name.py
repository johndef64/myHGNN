#%%

import requests
import json

def get_uniprot_entry_names(accession_list):
    # Endpoint per la ricerca
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    results = {}

    print(f"Inizio recupero per {len(accession_list)} accession(s)...\n")

    for acc in accession_list:
        # Chiediamo specificamente l'accession e l'id (Entry Name)
        params = {
            'query': f'accession:{acc}',
            'fields': 'accession,id',
            'format': 'json'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                # 'primaryAccession' è l'ID (Q88K39)
                # 'uniProtkbId' è l'Entry Name (ASPQ_PSEPK)
                entry_data = data['results'][0]
                entry_name = entry_data['uniProtkbId']
                
                results[acc] = entry_name
                print(f"✅ {acc} -> {entry_name}")
            else:
                results[acc] = "NOT_FOUND"
                print(f"❌ {acc} non trovato.")
                
        except Exception as e:
            print(f"⚠️ Errore per {acc}: {e}")
            results[acc] = "ERROR"

    return results

# --- CONFIGURAZIONE ---
# Inserisci qui la tua lista di Accession Numbers
mie_proteine = ["Q88K39", "P01112", "P62158", "P04637"]

"""

path = "dataset/DRUGBANK/pathogenkg_drug_target_triples.csv"

head,interaction,tail,source,type
Compound::DrugBank:DB15954,TARGET,Gene::Uniprot:P48167,DRUGBANK,Compound-Gene
Compound::DrugBank:DB15954,TARGET,Gene::Uniprot:O75311,DRUGBANK,Compound-Gene
Compound::DrugBank:DB15954,TARGET,Ge

get all the unique UniProt IDs in the dataset making the right parsing
"""
path = "dataset/DRUGBANK/pathogenkg_drug_target_triples.csv"
import pandas as pd
df = pd.read_csv(path, on_bad_lines='skip')

# Extract unique UniProt IDs from the 'tail' column (Gene::Uniprot:... / ExtGene::Uniprot:...)
unique_uniprot_ids = (
    df['tail']
    .astype(str)
    .str.strip()
    .str.extract(r'(?:Gene|ExtGene)::Uniprot:([A-Za-z0-9-]+)', expand=False)
    .dropna()
    .drop_duplicates()
    .tolist()
)
print(f"Unique UniProt IDs found: {len(unique_uniprot_ids)}")
#%%
# Esecuzione
mappa_nomi = get_uniprot_entry_names(unique_uniprot_ids)

# Salvataggio in JSON
filename = "dataset/DRUGBAK/uniprot_entry_names.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(mappa_nomi, f, indent=4)

print(f"\nFatto! Risultati salvati in '{filename}'")
