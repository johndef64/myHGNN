import random
import requests
from tqdm import tqdm
from collections import defaultdict
import os
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

os.chdir("dataset")

########################## Get Unprot id from taxonomyid ###################################

taxon_uniprot_ids = {}
# taxonomy_ids = ["9606"]  #,'418103'
taxonomy_ids = [
    "1423",   # Bacillus subtilis
    # "9606",
    # 562,    # Escherichia coli
    # "1280",   # Staphylococcus aureus
    # "287",    # Pseudomonas aeruginosa
    # 28901,  # Salmonella enterica
    # 210     # Helicobacter pylori
]

for taxonomy_id in tqdm(taxonomy_ids):
    # Your SPARQL query (as a string)
    sparql_query = """
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX up: <http://purl.uniprot.org/core/>
    PREFIX taxon: <http://purl.uniprot.org/taxonomy/>

    SELECT DISTINCT ?protein ?uniprotId
    WHERE {
    ?protein a up:Protein ;
            up:organism ?taxon ;
            up:mnemonic ?uniprotId .
    
    ?taxon up:scientificName ?scientificName .
    
    FILTER(?taxon = taxon:"""+str(taxonomy_id)+""")
    }
    ORDER BY ?uniprotId
    """

    # UniProt SPARQL endpoint
    url = "https://sparql.uniprot.org/sparql"

    # Set parameters
    params = {
        "query": sparql_query,
        "format": "json"   # or "tsv", "xml", etc.
    }

    # Send the request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise error if request failed

    # Parse the response
    results = response.json()

    # Print out the results
    # for binding in results["results"]["bindings"]:
    #     print(f"{binding['uniprotId']['value']}: {binding['protein']['value']}")
    
    data = []
    for binding in tqdm(results["results"]["bindings"]):
        #uniprot_id = binding['uniprotId']['value']
        protein_id = binding['protein']['value'].split('/')[-1]  # Extract the last part of the URI
        #data.append((uniprot_id, protein_id))
        data.append(protein_id)
    taxon_uniprot_ids[taxonomy_id] = data

# import json
# # Save the dictionary to a JSON file
# with open('taxon_uniprot_ids.json', 'w') as f:
#     json.dump(taxon_uniprot_ids, f, indent=4)

# # Load the dictionary from a JSON file
# with open('taxon_uniprot_ids.json', 'r') as f:
#     taxon_uniprot_ids = json.load(f)

# taxon_uniprot_ids.keys()  # Check the keys in the dictionary
# taxon_uniprot_ids[taxonomy_ids[0]]

########################## Get crossrefs from Uniprotids ###################################
n = 0 

dbs = ['EMBL',
       #'PIR', 'RefSeq',
       'AlphaFoldDB',
       #'SMR',
       'STRING',
       'PaxDb',
       'GeneID',
       'KEGG',
       #'CTD',
       'eggNOG', #'HOGENOM', 'InParanoid', 'OMA',
       'OrthoDB', #'TreeFam', 'Proteomes',
       'GO', #'CDD', FunFam', 'Gene3D',
       'InterPro',
       'PANTHER',
       'Pfam', #'PRINTS', 'SUPFAM', 'PROSITE'
       ]

## get funstion


# Lista di User-Agent comuni
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
]

def get_cross_references(uniprot_id, target_dbs):
    # Sleep randomico tra 1 e 3 secondi
    #time.sleep(random.uniform(0.001, 0.10))
    
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Warning: failed to retrieve data for {uniprot_id}")
        return {}

    data = response.json()
    xrefs = data.get('uniProtKBCrossReferences', [])

    db_dict = defaultdict(list)
    for xref in xrefs:
        db = xref.get("database")
        if db in target_dbs:
            db_dict[db].append(xref.get("id"))

    return db_dict

all_data = []
for taxonomy_id in taxonomy_ids:
    uniprot_ids = taxon_uniprot_ids[taxonomy_id] #[:10] # ["P69905", "P68871", "P12345"]  # Replace with your list
    target_dbs = dbs          # Replace with the databases you want
    for uid in tqdm(uniprot_ids[:]):
        db_refs = get_cross_references(uid, target_dbs)
        row = {"UniProtID": uid, "taxonomy_id": taxonomy_ids[n]}  # Add taxonomy ID to the row
        for db in target_dbs:
            row[db] = db_refs.get(db, [])
        all_data.append(row)

output_file = f'{taxonomy_ids[0]}_uniprot_crossref.json'
with open(output_file, 'w') as fout:
    for row in all_data:
        fout.write(row)