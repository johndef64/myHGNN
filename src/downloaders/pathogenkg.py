
#%%
import pandas as pd
import gzip
# from Bio import SwissProt
import os
print(os.getcwd())
#%%

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

path = r"dataset\drkg\drkg.zip"

drkg = pd.read_table(path,  low_memory=False)
#%%
drkg
drkg[drkg["type"] == "Gene-Compound"]#.head(10)
#%%
drkg[drkg["type"] == "Compound-Gene"]#.head(10)

#%%
# Print a better formatted report
report = {
    "interaction": drkg["interaction"].value_counts(),
    "source": drkg["source"].value_counts(),
    "type": drkg["type"].value_counts(),
}

for key, value in report.items():
    print(f"\n=== {key.upper()} ===")
    print(value.to_string())
# %%

drkg_gene = drkg[drkg["type"] == "Gene-Gene"]
print(drkg_gene['interaction'].value_counts())
len(drkg_gene)
# %%
print(drkg_gene['source'].value_counts())

# %%
drkg_string = drkg[drkg["source"] == "STRING"]
print(drkg_string['interaction'].value_counts())
print(len(drkg_string))
drkg_string
# %%
drkg_string[drkg_string["head"].str.contains(":22888")]
# %%
drkg[drkg["head"].str.contains(":22888")]

#%%

# GREAT CONVERSION OF STRING DATASET TO TRIPLES
import pandas as pd
alias_file = r"G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\STRING\9606.protein.aliases.v12.0.txt.gz"
h_string_file = r"G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\STRING\9606.protein.physical.links.v12.0.txt.gz"

h_string = pd.read_table(h_string_file, sep=" ", compression='gzip', low_memory=False)
h_alias_full = pd.read_table(alias_file, compression='gzip')
#%%

h_alias_full[h_alias_full[h_alias_full.columns[0]].str.contains("ENSP00000430774")]
#%%
all_genes= h_alias_full[h_alias_full.columns[0]].drop_duplicates()
all_genes
#%%
# if not contains "entrez" then use "KEGG_GENEID"
h_alias_full["source"].replace("KEGG_GENEID", "entrez", inplace=True)
h_alias_full["source"].replace("UniProt_DR_GeneID", "entrez", inplace=True)
h_alias_full["source"].replace("Ensembl_HGNC_entrez_id", "entrez", inplace=True)
h_alias = h_alias_full[h_alias_full["source"].str.contains("entrez")].reset_index(drop=True)
h_alias = h_alias.drop_duplicates()
#%%
#%%
from tqdm import tqdm
# Creare il dataset di triple
triples = []

count = 0
for _, row in tqdm(h_string.iterrows()):
    string_id1 = row['protein1']
    string_id2 = row['protein2']
    # print(string_id1, string_id2)
    alt_id1 = f'Gene::STRING:{string_id1.split(".")[0]}'
    alt_id2 = f'Gene::STRING:{string_id2.split(".")[0]}'
    alias1 = h_alias[h_alias['#string_protein_id'] == string_id1]['alias'].values[0] if not h_alias[h_alias['#string_protein_id'] == string_id1].empty else alt_id1
    alias2 = h_alias[h_alias['#string_protein_id'] == string_id2]['alias'].values[0] if not h_alias[h_alias['#string_protein_id'] == string_id2].empty else alt_id2

    triple = {
        'head': f'Gene::NCBI:{alias1}',
        'interaction': 'PHYSICAL',
        'tail': f'Gene::NCBI:{alias2}',
        'source': 'STRING',
        'type': 'Gene-Gene'
    }
    triples.append(triple)
    count += 1
    if count  == 10:
         break

# Convertire in DataFrame
triples_df = pd.DataFrame(triples)

# Salvare il dataset di triple (opzionale)
triples_df#.to_csv('pathogenkg_drug_target_triples.csv', index=False)
#%% MULTI THREADING
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def process_chunk(chunk_data):
    """
    Processa un chunk di righe del dataset
    """
    chunk, h_alias = chunk_data
    triples = []
    
    for _, row in chunk.iterrows():
        string_id1 = row['protein1']
        string_id2 = row['protein2']
        
        alt_id1 = f'{string_id1.split(".")[1]}'
        alt_id2 = f'{string_id2.split(".")[1]}'
        
        # Lookup degli alias
        alias1_match = h_alias[h_alias['#string_protein_id'] == string_id1]['alias'].values
        alias2_match = h_alias[h_alias['#string_protein_id'] == string_id2]['alias'].values
        
        alias1 = alias1_match[0] if len(alias1_match) > 0 else alt_id1
        alias2 = alias2_match[0] if len(alias2_match) > 0 else alt_id2

        triple = {
            'head': f'Gene::NCBI:{alias1}',
            'interaction': 'PHYSICAL',
            'tail': f'Gene::NCBI:{alias2}',
            'source': 'STRING',
            'type': 'Gene-Gene'
        }
        triples.append(triple)
    
    return triples

def create_triples_multithread(h_string, h_alias, n_threads=4, chunk_size=None):
    # Calcola la dimensione del chunk se non specificata
    if chunk_size is None:
        chunk_size = max(1, len(h_string) // (n_threads * 4))
    
    # Dividi il dataset in chunk
    chunks = []
    for i in range(0, len(h_string), chunk_size):
        chunk = h_string.iloc[i:i+chunk_size]
        chunks.append((chunk, h_alias))
    
    print(f"Processando {len(h_string)} righe in {len(chunks)} chunk usando {n_threads} thread...")
    
    all_triples = []
    
    # Processa i chunk in parallelo
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Invia tutti i job
        future_to_chunk = {executor.submit(process_chunk, chunk_data): i 
                          for i, chunk_data in enumerate(chunks)}
        
        # Raccogli i risultati con progress bar
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_triples = future.result()
                    all_triples.extend(chunk_triples)
                    pbar.update(1)
                except Exception as exc:
                    print(f'Chunk {chunk_idx} ha generato un\'eccezione: {exc}')
                    pbar.update(1)
    
    # Convertire in DataFrame
    triples_df = pd.DataFrame(all_triples)
    print(f"Processate {len(triples_df)} triple")
    
    return triples_df

# Versione alternativa con ottimizzazioni per lookup degli alias
def create_triples_multithread_optimized(h_string, h_alias, n_threads=4, chunk_size=None):
    """
    Versione ottimizzata che pre-processa gli alias in un dizionario per lookup più veloci
    """
    
    # Pre-processare gli alias in un dizionario per lookup O(1)
    print("Pre-processando gli alias...")
    alias_dict = h_alias.set_index('#string_protein_id')['alias'].to_dict()
    
    def process_chunk_optimized(chunk_data):
        chunk, alias_lookup = chunk_data
        triples = []
        
        for _, row in chunk.iterrows():
            string_id1 = row['protein1']
            string_id2 = row['protein2']
            
            alt_id1 = f'{string_id1.split(".")[1]}'
            alt_id2 = f'{string_id2.split(".")[1]}'
            
            # Lookup O(1) usando il dizionario
            alias1 = alias_lookup.get(string_id1, alt_id1)
            alias2 = alias_lookup.get(string_id2, alt_id2)

            triple = {
                'head': f'Gene::NCBI:{alias1}',
                'interaction': 'PHYSICAL',
                'tail': f'Gene::NCBI:{alias2}',
                'source': 'STRING',
                'type': 'Gene-Gene'
            }
            triples.append(triple)
        
        return triples
    
    # Calcola la dimensione del chunk se non specificata
    if chunk_size is None:
        chunk_size = max(1, len(h_string) // (n_threads * 4))
    
    # Dividi il dataset in chunk
    chunks = []
    for i in range(0, len(h_string), chunk_size):
        chunk = h_string.iloc[i:i+chunk_size]
        chunks.append((chunk, alias_dict))
    
    print(f"Processando {len(h_string)} righe in {len(chunks)} chunk usando {n_threads} thread...")
    
    all_triples = []
    
    # Processa i chunk in parallelo
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Invia tutti i job
        future_to_chunk = {executor.submit(process_chunk_optimized, chunk_data): i 
                          for i, chunk_data in enumerate(chunks)}
        
        # Raccogli i risultati con progress bar
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_triples = future.result()
                    all_triples.extend(chunk_triples)
                    pbar.update(1)
                except Exception as exc:
                    print(f'Chunk {chunk_idx} ha generato un\'eccezione: {exc}')
                    pbar.update(1)
    
    # Convertire in DataFrame
    triples_df = pd.DataFrame(all_triples)
    print(f"Processate {len(triples_df)} triple")
    
    return triples_df

# Esempio di utilizzo:
if __name__ == "__main__":

    # Versione base multi-thread
    # triples_df = create_triples_multithread(h_string, h_alias, n_threads=10)
    
    # Versione ottimizzata (raccomandata per dataset grandi)
    triples_df = create_triples_multithread_optimized(h_string, h_alias, n_threads=10)
    
    # Salvare il dataset (opzionale)
    triples_df.to_csv('human_string_triples.csv', index=False)
    
    pass
#%%
import pandas as pd
triples_df = pd.read_csv('human_string_triples.zip', low_memory=False)


filtered_df = triples_df[
    triples_df["head"].str.contains("ENS") | triples_df["tail"].str.contains("ENS")
]
filtered_df["head"].nunique(), filtered_df["tail"].nunique()

#%%
# triples_df = pd.DataFrame(triples)
d1  =triples_df[triples_df["head"].str.contains(":22888")]
d2 = triples_df[triples_df["head"].str.contains(":84993")]

d1ind2 = d1[d1["tail"].isin(d2["head"])]

d2ind1 = d2[d2["tail"].isin(d1["head"])]
d1ind2, d2ind1
drkg_string["head", "tail"]
#%%


drkg_string_small = drkg_string[["head", "tail"]]#.drop_duplicates()
triples_df_small = triples_df[["head", "tail"]].drop_duplicates()
len(drkg_string_small), len(triples_df_small)

#check how many triples are in drkg_string that are not in triples_df_small
missing_triples = drkg_string_small[~drkg_string_small.set_index(['head', 'tail']).index.isin(triples_df_small.set_index(['head', 'tail']).index)]
print(f"Missing triples in drkg_string: {len(missing_triples)}")
#check how many triples are in triples_df_small that are not in drkg_string_small
missing_triples_df = triples_df_small[~triples_df_small.set_index(['head', 'tail']).index.isin(drkg_string_small.set_index(['head', 'tail']).index)]
print(f"Missing triples in triples_df_small: {len(missing_triples_df)}")
#%%
# Remove reverse duplicates
drkg_string_small_rev = drkg_string_small.drop_duplicates(subset=["head", "tail"])
len(drkg_string_small_rev)
#%%


import pandas as pd
# Function to remove reverse duplicates
def remove_reverse_duplicates(df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    # Create an auxiliary column with sorted tuples
    df['sorted_pair'] = df.apply(lambda row: tuple(sorted([row['head'], row['tail']])), axis=1)
    # Remove reverse duplicates
    df_noreverse = df.drop_duplicates(subset='sorted_pair', keep='first')
    # Reset index
    df_noreverse = df_noreverse.reset_index(drop=True)
    # Remove auxiliary column
    df_noreverse = df_noreverse.drop('sorted_pair', axis=1)

    return df_noreverse
# Applichiamo la funzione
string_noreverse = remove_reverse_duplicates(triples_df_small)
print("Numero di triple senza duplicati inversi in STRING:", len(string_noreverse), "percentuale:", len(string_noreverse) / len(triples_df_small) * 100)
#%%
drkg_string_noreverse = remove_reverse_duplicates(drkg_string_small)
print("Numero di triple senza duplicati inversi in VITA (STRING):", len(drkg_string_noreverse), "percentuale:", len(drkg_string_noreverse) / len(drkg_string_small) * 100)
#%%
drkg_small = drkg[["head","tail"]].drop_duplicates()
drkg_noreverse = remove_reverse_duplicates(drkg_small)
print("Numero di triple senza duplicati inversi in VITA (all):", len(drkg_noreverse), "percentuale:", len(drkg_noreverse) / len(drkg_small) * 100)

#%%
len(drkg)


#%%
drkg_string_small = drkg_string[["head", "interaction","tail"]].reset_index(drop=True)
drkg_string_small[drkg_string_small.duplicated()].sort_values(by=["head", "tail"])
#%%
drkg_string_small.drop_duplicates()
# %%

drkg_drug = drkg[drkg["source"] == "DRUGBANK"]
print(drkg_drug['interaction'].value_counts())

drkg_drug[drkg_drug["interaction"] == "ENZYME"].head(10).to_clipboard()

# %%

drkg[drkg["source"] == "STRING"].head(10)
# %%
drkg

