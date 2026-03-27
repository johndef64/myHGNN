#%%
import glob

import requests
import os
from tqdm import tqdm
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
os.chdir('dataset')

# Txonomy ID list (version 1)
tax_ids_prev =[9606, 83332, 224308, 208964, 99287, 71421, 243230, 
          85962, 171101, 243277, 294, 1314, 272631,
          212717, 36329, 237561, 6183, 5664, 185431, 330879]
 # ad esempio, 9606 = Homo sapiens, 294 = Saccharomyces cerevisiae

# load all taxa ids (version 2)
import pandas as pd
taxa_df = pd.read_csv("DRUGBANK/taxons/drugbank_string_taxa_merged_string_with_pathogen_status.csv")
pathogen_taxa = taxa_df[taxa_df.human_pathogen != 'No']
non_pathogen_taxa = taxa_df[taxa_df.human_pathogen == 'No']
tax_ids = pathogen_taxa.taxonomy_id.to_list()
print(f"Preparing download for {len(tax_ids)} pathogen taxa")
# tax_ids = list(set(tax_ids) - set(tax_ids_prev))

#%%

# get all downloaded taxa id from files.gz in dataset/STRING  
# split by "." and take first part
import glob

downloaded_taxa = []
print(os.getcwd())
for file in glob.glob("STRING/*.txt.gz"):
    filename = os.path.basename(file)
    tax_id = filename.split(".")[0]
    downloaded_taxa.append(tax_id)
    downloaded_taxa = list(set(downloaded_taxa))
print(f"Already downloaded taxa IDs len {len(downloaded_taxa)}")

# remove downloadded from tax_ids
tax_ids = [tax_id for tax_id in tax_ids if str(tax_id) not in downloaded_taxa]
print(f"Taxa IDs to download len {len(tax_ids)}")
#%%

# Lista dei file da scaricare per ogni tax_id
files = [
    {
        'base_url': 'https://stringdb-downloads.org/download/',
        'subdir': 'protein.links.v12.0',
        'suffix': 'protein.links.v12.0'
    },
    {
        'base_url': 'https://stringdb-downloads.org/download/',
        'subdir': 'protein.physical.links.v12.0',
        'suffix': 'protein.physical.links.v12.0'
    },
    {
        'base_url': 'https://stringdb-downloads.org/download/',
        'subdir': 'protein.aliases.v12.0',
        'suffix': 'protein.aliases.v12.0'
    },
    {
        'base_url': 'https://stringdb-downloads.org/download/',
        'subdir': 'protein.enrichment.terms.v12.0',
        'suffix': 'protein.enrichment.terms.v12.0'
    },
    {
        'base_url': 'https://stringdb-downloads.org/download/',
        'subdir': 'protein.orthology.v12.0',
        'suffix': 'protein.orthology.v12.0'
    },
    {
        'base_url': 'https://stringdb-static.org/download/',
        'subdir': 'protein.actions.v11.0',
        'suffix': 'protein.actions.v11.0'
    }
]

def download_string_files(tax_id, file_info, output_folder):
    """Scarica e salva un file da STRING database in una cartella specifica."""
    base_url = file_info['base_url']
    subdir = file_info['subdir']
    suffix = file_info['suffix']

    url = f"{base_url}{subdir}/{tax_id}.{suffix}.txt.gz"

    try:
        # Crea la cartella di output se non esiste
        os.makedirs(output_folder, exist_ok=True)

        # Costruisci il nome del file di output
        filename = f"{tax_id}.{suffix}.txt.gz"
        output_path = os.path.join(output_folder, filename)

        # Invia una richiesta HTTP con un user-agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Solleva un'eccezione per codici di errore HTTP

        # Scrivi il contenuto della risposta nel file
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File scaricato e salvato in: {output_path}")
        return True

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP durante il download di {url}: {http_err}")
    except Exception as err:
        print(f"Errore generico durante il download di {url}: {err}")
    return False

# Specifica la cartella in cui vuoi salvare i file (ad esempio, "string_files")
output_folder = "STRING"

# Esegui il download per ogni tax_id e ogni file

for tax_id in tqdm(tax_ids):
    tax_id = str(tax_id)  # Assicurati che tax_id sia una stringa
    print(f"\nInizio download per tax_id: {tax_id}")
    for file_info in files:
        download_string_files(tax_id, file_info, output_folder)
    print(f"Download completati per tax_id: {tax_id}\n")

print(f"\nTutti i file sono stati salvati nella cartella: {os.path.abspath(output_folder)}")
# %%
