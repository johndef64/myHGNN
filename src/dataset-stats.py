#%%
import os
import logging
import argparse
import zipfile
import json
from time import time
from collections import defaultdict
import pandas as pd

minimal_available_targets = [
    '83332', '224308', '208964', '99287', '71421', '243230',
    '85962', '171101', '243277', '294', '1314', '272631',
    '212717', '36329', '237561', '6183', '5664', '185431', '330879'
]
# convert to integer taxonomy IDs
minimal_available_targets_int = [int(tax_id) for tax_id in minimal_available_targets]


import pandas as pd
taxa_df = pd.read_csv("../dataset/DRUGBANK/taxons/drugbank_string_taxa_merged_string_with_pathogen_status.csv")

taxa_df[taxa_df['STRING_type'] == "core"].domain.value_counts()
taxa_df[taxa_df['STRING_type'] == "periphery"].domain.value_counts()
# taxa_df

taxa_df.human_pathogen.value_counts()
taxa_df = taxa_df[taxa_df.domain == 'Bacteria']
pathogen_taxa = taxa_df[taxa_df.human_pathogen != 'No']
non_pathogen_taxa = taxa_df[taxa_df.human_pathogen == 'No']
pathogen_taxa_core = pathogen_taxa[pathogen_taxa['STRING_type'] == 'core']
pathogen_taxa_periphery = pathogen_taxa[pathogen_taxa['STRING_type'] == 'periphery']
pathogen_taxa.STRING_type.value_counts() 

# print len of all the sets
print(f"Total taxa: {len(taxa_df)}")
print(f"Pathogen taxa: {len(pathogen_taxa)}")
print(f"Non-pathogen taxa: {len(non_pathogen_taxa)}")
print(f"Pathogen taxa (core): {len(pathogen_taxa_core)}")
print(f"Pathogen taxa (periphery): {len(pathogen_taxa_periphery)}")

#%%
# Pretty print pathogen_taxa_core without index and STRING_type
display_cols = [c for c in pathogen_taxa_core.columns if c not in  ["official_name_NCBI","STRING_name_compact",'STRING_type', "domain"]]
print("\n=== Pathogen Taxa (core) ===")
print(pathogen_taxa_core[display_cols].to_csv(sep='\t', index=False))

# Statistics
print(f"\n=== Statistics ===")
print(f"Rows: {len(pathogen_taxa_core)}")
print(f"Columns: {display_cols}")
print(f"\nDomain distribution:\n{pathogen_taxa_core.domain.value_counts().to_string()}")
print(f"\nHuman pathogen status:\n{pathogen_taxa_core.human_pathogen.value_counts().to_string()}")
print(f"\nUnique taxonomy IDs: {pathogen_taxa_core.taxonomy_id.nunique()}")

print("uniptot_count stands for the UniprotId if the protein that is taget of a Drug in Drugbank")


# Reload the merged dataset and analyze only TARGET triples (Compound-ExtGene)

merged_path = os.path.join("../dataset", f'PathogenKG_n31_core.tsv.zip')
print(f"Loading merged KG from: {merged_path}")
kg_df = pd.read_csv(merged_path, sep='\t', compression='zip', dtype=str)

print(f"\nTotal triples: {len(kg_df)}")
print(f"\nTriple types distribution:\n{kg_df['interaction'].value_counts().to_string()}")

# Filter only drug-target triples
target_triples = kg_df[kg_df['interaction'] == 'TARGET'].copy()
print(f"\n=== TARGET triples (Compound-ExtGene) ===")
print(f"Total drug-target triples: {len(target_triples)}")
#%%
# Unique drugs and gene targets
unique_drugs = target_triples['head'].nunique()
unique_genes = target_triples['tail'].nunique()
unique_interactions = target_triples['interaction'].nunique()
print(f"Unique drugs (Compound): {unique_drugs}")
print(f"Unique gene targets (ExtGene): {unique_genes}")
print(f"Unique interaction types: {unique_interactions}")

# Interaction type breakdown
print(f"\nInteraction types:\n{target_triples['interaction'].value_counts().to_string()}")

# Source breakdown
print(f"\nSources:\n{target_triples['source'].value_counts().to_string()}")

# Top drugs by number of targets
drug_target_counts = target_triples.groupby('head')['tail'].nunique().sort_values(ascending=False)
print(f"\nTop 20 drugs by number of gene targets:")
print(drug_target_counts.head(20).to_string())

# Top gene targets by number of drugs
gene_drug_counts = target_triples.groupby('tail')['head'].nunique().sort_values(ascending=False)
print(f"\nTop 20 gene targets by number of drugs:")
print(gene_drug_counts.head(20).to_string())
#%%
"""
Top 20 drugs by number of gene targets:
head
Compound::Pubchem:87642       8
Compound::Pubchem:643976      7
Compound::Pubchem:643975      6
Compound::Pubchem:64689       5

"""
# %%
# dammi uno script che recupra i PubChemID da drug_target_counts e recuptera da PubChem con lapi python i nomi dei composti ed info che sono riulevati per il mio deataset, non troppe solo quelle rilvanti, e poi stampa una tabella con PubChemID, nome composto, numero di target


#%%
"""Fetch compound info from PubChem for the drugs in the merged PathogenKG."""

import time
import requests
import pandas as pd

# --- 1. Load the merged KG and extract drug-target triples ---
merged_path = 'dataset/PathogenKG_n31_core.tsv.zip'
kg_df = pd.read_csv(merged_path, sep='\t', compression='zip', dtype=str)
target_triples = kg_df[kg_df['type'] == 'Compound-ExtGene']

# Count targets per drug
drug_target_counts = target_triples.groupby('head')['tail'].nunique().sort_values(ascending=False)
drug_target_counts
#%%
# Extract PubChem CIDs
cid_map = {}  # cid_str -> original head label
for head_label in drug_target_counts.index:
    # "Compound::Pubchem:87642" -> "87642"
    cid = head_label.split('Pubchem:')[-1]
    cid_map[cid] = head_label
cid_map
#%%
cids = list(cid_map.keys())
print(f"Total unique drugs to query: {len(cids)}")

# --- 2. Query PubChem PUG REST in batches ---
BATCH_SIZE = 50  # PubChem allows up to ~100 CIDs per request
PROPERTIES = 'IUPACName,MolecularFormula,MolecularWeight,IsomericSMILES'

results = []
for i in range(0, len(cids), BATCH_SIZE):
    batch = cids[i:i + BATCH_SIZE]
    cid_list = ','.join(batch)
    url = (
        f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}'
        f'/property/{PROPERTIES}/JSON'
    )
    print(f"Fetching batch {i // BATCH_SIZE + 1} ({len(batch)} CIDs)...")
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        props = resp.json().get('PropertyTable', {}).get('Properties', [])
        results.extend(props)
    else:
        print(f"  WARNING: HTTP {resp.status_code} for batch starting at index {i}")
    time.sleep(0.3)  # be polite to PubChem

# Also fetch synonyms (first synonym = common name) in batches
synonyms_map = {}
for i in range(0, len(cids), BATCH_SIZE):
    batch = cids[i:i + BATCH_SIZE]
    cid_list = ','.join(batch)
    url = (
        f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}'
        f'/synonyms/JSON'
    )
    print(f"Fetching synonyms batch {i // BATCH_SIZE + 1}...")
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        info_list = resp.json().get('InformationList', {}).get('Information', [])
        for info in info_list:
            cid_str = str(info.get('CID', ''))
            syns = info.get('Synonym', [])
            synonyms_map[cid_str] = syns[0] if syns else ''
    else:
        print(f"  WARNING: HTTP {resp.status_code} for synonyms batch at index {i}")
    time.sleep(0.3)
#%%
# --- 3. Build the final table ---
rows = []
for item in results:
    cid_str = str(item['CID'])
    head_label = cid_map.get(cid_str, '')
    n_targets = int(drug_target_counts.get(head_label, 0))
    rows.append({
        'PubChemCID': cid_str,
        'Name': synonyms_map.get(cid_str, ''),
        'MolecularFormula': item.get('MolecularFormula', ''),
        'MolecularWeight': item.get('MolecularWeight', ''),
        'IUPAC': item.get('IUPACName', ''),
        'SMILES': item.get('IsomericSMILES', ''),
        'NumTargets': n_targets,
    })

df_drugs = pd.DataFrame(rows).sort_values('NumTargets', ascending=False).reset_index(drop=True)

# --- 4. Print ---
print(f"\n{'='*90}")
print(f"Drug compounds in PathogenKG  (total: {len(df_drugs)})")
print(f"{'='*90}")
display_df = df_drugs[['PubChemCID', 'Name', 'MolecularFormula', 'MolecularWeight', 'NumTargets']]
print(display_df.to_csv(sep='\t', index=False))

# Save to CSV
out_csv = 'dataset/pathogenkg_drug_info.csv'
df_drugs.to_csv(out_csv, index=False)
print(f"\nFull table saved to {out_csv}")