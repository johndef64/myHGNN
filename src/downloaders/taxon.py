import os

import pandas as pd
from Bio import Entrez
import time
    
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

# Imposta il tuo indirizzo email per Entrez 
Entrez.email = "your.email@example.com"  

# Carico il dataframe
df = pd.read_csv('proteins_uniprot_id_to_taxonomy_id_counts.csv')

# Creo una nuova colonna per il nome scientifico
df['scientific_name'] = None

# Funzione per ottenere il nome scientifico dal taxonomy ID
def get_scientific_name(tax_id):
    try:
        # Usa esummary invece di efetch per ottenere informazioni sulla tassonomia
        handle = Entrez.esummary(db='taxonomy', id=str(tax_id))
        record = Entrez.read(handle)[0]
        handle.close()
        return record['ScientificName']
    except Exception as e:
        print(f"Errore per taxonomy ID {tax_id}: {e}")
        return None

# Popolo la nuova colonna con gestione degli errori e rate limiting
for index, row in df.iterrows():
    tax_id = row['taxonomy_id']
    print(f"Processando riga {index + 1}/{len(df)}: taxonomy_id = {tax_id}")
    
    scientific_name = get_scientific_name(tax_id)
    df.loc[index, 'scientific_name'] = scientific_name
    
    # Rate limiting per evitare di sovraccaricare il server NCBI
    time.sleep(0.5)  # Pausa di 0.5 secondi tra le richieste

# Salvo il dataframe aggiornato
df#.to_csv('drugbank_uniprot_id_to_taxonomy_id_counts_with_scientific_names.csv', index=False)
print("File salvato con successo!")

# Mostra statistiche finali
print(f"\nStatistiche:")
print(f"Totale righe processate: {len(df)}")
print(f"Nomi scientifici trovati: {df['scientific_name'].notna().sum()}")
print(f"Nomi scientifici mancanti: {df['scientific_name'].isna().sum()}")


 # %%
df.to_csv('proteins_uniprot_id_to_taxonomy_id_counts.csv', index=False)
df


#%%
####### ##########
# Merge DrugBank and STRING species data ##########
# https://version10.string-db.org/help/database/
# type	If the organism is a core species or periphery species. 
# Core species are BLAST aligned all-against-all, periphery only 
# against the core.

# https://www.researchgate.net/figure/Organisms-covered-by-STRING-STRING-currently-contains-373-fully-sequenced-organisms_fig3_6698455

string_species = pd.read_table('string_species.v12.0.txt')
string_species['#taxon_id']

drugbank_taxa = pd.read_csv('proteins_uniprot_id_to_taxonomy_id_counts.csv')

# Merge the two dataframes on the taxonomy ID
merged_df = pd.merge(drugbank_taxa, 
                     string_species, 
                     left_on='taxonomy_id', 
                     right_on='#taxon_id', 
                     how='left')

# rename columns for clarity
merged_df.rename(columns={'#uniprot_count': '#drugbank_targets_count', 
                           #'scientific_name': 'drugbank_scientific_name'
                           }, inplace=True)
merged_df = merged_df[['taxonomy_id', 'uniprot_count', 'scientific_name', 
       'STRING_type', 'STRING_name_compact', 'official_name_NCBI', 'domain']]
merged_df.to_csv('drugbank_string_taxa_merged.csv', index=False)
# %%

# remove line with no STING_type
merged_df_compact = merged_df[merged_df['STRING_type'].notna()]
merged_df_compact#.to_csv('drugbank_string_taxa_merged_compact.csv', index=False)


taxa_selected =[83332, 224308, 208964, 99287, 71421, 243230, 85962, 171101, 243277, 294, 1314, 272631, 212717, 36329, 237561, 6183, 5664, 185431, 330879]