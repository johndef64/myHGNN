#%%
import requests
import os
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
    
os.chdir("dataset")

# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/

# Correct organism name
organism = "viruses"

# Build paths and filenames
filename = f"uniprot_sprot_{organism}.dat.gz"
directory = "uniprot_sprot"
output_path = os.path.join(directory, filename)
url = f"https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/{filename}"

#%%
# Ensure the output directory exists
os.makedirs(directory, exist_ok=True)

if os.path.exists(output_path):
    print(f"{output_path} already exists. Skipping download.")
else:
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises HTTPError if the download failed
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        print(f"Downloaded and saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {e}")

#%%

import gzip
from Bio import SwissProt

with gzip.open("uniprot_sprot_fungi.dat.gz", "rt") as handle:
    for record in SwissProt.parse(handle):
        print(record.entry_name, record.description)
        # Process each record as needed

# %%

import gzip
from Bio import SwissProt
import pandas as pd
from tqdm import tqdm

records = []
organism = "human" # Cambia in "fungi", "bacteria", "viruses", etc. se necessario
directory = "uniprot_sprot"
filename = f"uniprot_sprot_{organism}.dat.gz" 
output_path = os.path.join(directory, filename)
#%%
# with gzip.open(filename, "rt") as handle:
#     for record in tqdm(SwissProt.parse(handle)):
#         # Estraggo le info principali
#         entry_name = record.entry_name
#         accession = record.accessions[0] if record.accessions else None
#         protein_name = record.description
#         organism = record.organism
#         sequence = record.sequence
        
#         records.append({
#             "entry_name": entry_name,
#             "accession": accession,
#             "protein_name": protein_name,
#             "organism": organism,
#             "sequence": sequence
#         })

with gzip.open(output_path, "rt") as handle:
    for record in tqdm(SwissProt.parse(handle)):
        # Estrai tutti gli attributi pubblici (non quelli interni "__" o callable)
        record_dict = {
            attr: getattr(record, attr)
            for attr in dir(record)
            if not attr.startswith("_") and not callable(getattr(record, attr))
        }
        records.append(record_dict)


# Creo il DataFrame
df = pd.DataFrame(records)

# Mostra le prime righe
print(df.head())
df
#%%
print(df.columns)
df.head(100)
#%%
df.to_csv(output_path.replace(".dat", ".csv"), index=False, compression='gzip')

# %%
import ast
df = pd.read_csv(output_path.replace(".dat", ".csv"), compression='gzip')
df['accessions'] = df['accessions'].apply(ast.literal_eval)
df_exploded = df.explode('accessions')
df_exploded


# %%
import pandas as pd
import ast


######  PARSE CROSSREFERENCES ######
organism = "human"  # Cambia in "fungi", "bacteria", etc. se necessario
filename = f"uniprot_sprot_{organism}.csv.gz" 
path = os.path.join(directory, filename)
df = pd.read_csv(path, compression='gzip')
mappings = df[['accessions', "cross_references", 'taxonomy_id']]#.drop_duplicates()
for col in mappings.columns:
    mappings[col] = mappings[col].apply(ast.literal_eval)
mappings
#%%
mappings = mappings.explode('accessions').reset_index(drop=True)
mappings = mappings.explode('taxonomy_id').reset_index(drop=True)
mappings = mappings.explode("cross_references").reset_index(drop=True)
mappings
#%%
print(mappings.head().to_csv())

m = mappings.copy()
m = m[m['cross_references'].apply(lambda x: isinstance(x, tuple))]

# 1) ora puoi estrarre db_name e db_ids in sicurezza
m['db_name'] = m['cross_references'].apply(lambda tpl: tpl[0])
m['db_ids']  = m['cross_references'].apply(lambda tpl: list(tpl[1:]))

# 2) procedi con il pivot come prima
wide = (
    m
    .drop(columns='cross_references')
    .groupby(['accessions', 'taxonomy_id', 'db_name'])['db_ids']
    .agg(list)
    .unstack(fill_value=[])
    .reset_index()
)

# 3) rimuovi eventuali multi-indici residui
# wide.columns = ['accessions', 'taxonomy_id'] + list(wide.columns.levels[1][2:])
wide
#%%
wide.to_csv(f"uniprot_{organism}_crossref.csv.gz", index=False, compression='gzip')

#%%


# Parsing sicuro della colonna cross_references
def parse_cross_references(val):
    try:
        items = ast.literal_eval(val)
        if not items:
            return []
        # Se è una singola tupla, la metto in lista
        if isinstance(items, tuple):
            items = [items]
        return items
    except Exception:
        return []

mappings['cross_tuples'] = mappings['cross_references'].apply(parse_cross_references)

# Espando tutte le tuple in nuove righe (explode)
df_expanded = mappings.explode('cross_tuples')

# Elimino le righe senza cross_tuples
df_expanded = df_expanded[df_expanded['cross_tuples'].notnull()]

# Estraggo db e db_id
df_expanded['db'] = df_expanded['cross_tuples'].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None)
df_expanded['db_id'] = df_expanded['cross_tuples'].apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None)

# Raggruppo e aggrego i db_id come lista
pivot_df = (
    df_expanded
    .groupby(['accessions', 'taxonomy_id', 'db'])['db_id']
    .agg(list)
    .reset_index()
    .pivot(index=['accessions', 'taxonomy_id'], columns='db', values='db_id')
    .reset_index()
)
pivot_df
#%%
df = pivot_df.copy()
from tqdm import tqdm
for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].str.startswith('[').any():  
            if df[col].str.contains('nan').any():
                # Assicurati che la colonna sia di tipo stringa
                df[col] = df[col].apply(ast.literal_eval)
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
df.to_csv(f"uniprot_{organism}_crossref.csv.gz", index=False, compression='gzip')
df
#%%
# # Parsing della colonna cross_references
# df['cross_tuple'] = df['cross_references'].apply(ast.literal_eval)
# df['db'] = df['cross_tuple'].apply(lambda x: x[0])
# df['db_id'] = df['cross_tuple'].apply(lambda x: x[1])

# # Se ci sono più chiavi per riga, serve un identificatore unico per la pivot
# df['row_id'] = df.groupby(['accessions', 'taxonomy_id']).cumcount()

# # Pivot della tabella
# pivot_df = df.pivot_table(index=['accessions', 'taxonomy_id', 'row_id'], columns='db', values='db_id', aggfunc='first').reset_index()

# # Se vuoi togliere row_id se non serve più (ad esempio se c'è solo un set per accessions/taxonomy_id)
# pivot_df = pivot_df.drop(columns='row_id')


#%%
pivot_df.to_csv(f"uniprot_{organism}_crossref.csv.gz", index=False, compression='gzip')

#%%%
import pandas as pd
organism = "bacteria"  # Cambia in "fungi", "viruses", etc. se necessario
dff = pd.read_csv(f"uniprot_{organism}_crossref.csv.gz")
dff
#%%

dff[["accessions", "eggNOG"]].drop_duplicates().reset_index()


#%%
# Get taxon id Mappings

from Bio import Entrez

# Always set your email address when using Entrez
Entrez.email = "your_email@example.com"

def get_taxon_name_mapping(taxon_ids):
    mapping = {}
    # NCBI allows batch queries; join IDs with commas
    with Entrez.efetch(db="taxonomy", id=",".join(map(str, taxon_ids)), retmode="xml") as handle:
        records = Entrez.read(handle)
        for record in tqdm(records):
            tax_id = record["TaxId"]
            sci_name = record["ScientificName"]
            mapping[tax_id] = sci_name
    return mapping

# Example usage with a list of taxon IDs
taxon_ids = [9606, 10090, 7227]  # Homo sapiens, Mus musculus, Drosophila melanogaster
mapping = get_taxon_name_mapping(taxon_ids)
print(mapping)
