import requests
import time
import os

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

os.chdir("dataset")

def get_go_annotations(uniprot_ids):
    """
    Recupera le annotazioni GO per una lista di UniProt IDs.
    Restituisce un dizionario: {uniprot_id: [lista_GO_IDs]}
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    results = {}
    batch_size = 100  # UniProt REST API consente batch di max 500

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])
        params = {
            "query": query,
            "fields": "accession,go_id",
            "format": "tsv",
            "size": batch_size
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Errore per batch {batch}: {response.status_code}")
            continue

        lines = response.text.strip().split("\n")
        header = lines[0].split("\t")
        acc_idx = header.index("Entry")
        print("Header:", header)
        go_idx = header.index("Gene Ontology IDs")
        for line in lines[1:]:
            cols = line.split("\t")
            uniprot_id = cols[acc_idx]
            go_ids = cols[go_idx].split("; ") if cols[go_idx] else []
            results[uniprot_id] = go_ids
        time.sleep(1)  # Rispetta le policy API

    return results

# Esempio di utilizzo
uniprot_ids = ["P12345", "Q8N158", "O00327"]  # Sostituisci con i tuoi UniProt ID
go_annotations = get_go_annotations(uniprot_ids)

# Stampa risultati
for uid, go_ids in go_annotations.items():
    print(f"{uid}: {', '.join(go_ids)}")
