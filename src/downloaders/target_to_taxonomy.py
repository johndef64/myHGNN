import re
import os
import ast
import warnings
import requests
import pandas as pd
from tqdm.auto import tqdm

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

targets_path = 'drugbank_to_uniprot_targets_mapping.csv'
output_path = 'proteins_uniprot_id_to_taxonomy_id.csv'
dataset_path = 'dataset/'
crossref_paths = ['uniprot_bacteria_crossref.csv', 'uniprot_fungi_crossref.csv', \
                  'uniprot_human_crossref.csv', 'uniprot_viruses_crossref.csv', \
                  'uniprot_archaea_crossref.csv', 'uniprot_invertebrates_crossref.csv']

def uniprot_to_taxonomy_id(uniprot_id):
  try:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
      # Cerca il taxonomy ID nel testo della risposta
      for line in response.text.split('\n'):
        if line.startswith('OX   NCBI_TaxID='):
          tax_id = re.search(r'NCBI_TaxID=(\d+)', line)
          if tax_id:
            return tax_id.group(1)
                      
  except Exception as e:
    print(f"Errore nel parsing SwissProt: {e}")
  return None

def count_uniprot_by_taxonomy(csv_file):
    try:
        # Leggi il CSV
        df = pd.read_csv(csv_file)
        
        # Verifica che le colonne esistano
        required_columns = ['uniprot_id', 'taxonomy_id']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Il CSV deve contenere le colonne: {required_columns}")
        
        # Conta uniprot_id per taxonomy_id
        counts = df.groupby('taxonomy_id')['uniprot_id'].count().reset_index()
        counts.columns = ['taxonomy_id', 'uniprot_count']
        
        # Ordina per conteggio decrescente
        counts = counts.sort_values('uniprot_count', ascending=False)
        
        return counts
        
    except FileNotFoundError:
        print(f"Errore: File '{csv_file}' non trovato")
        return None
    except Exception as e:
        print(f"Errore: {e}")
        return None

if __name__ == '__main__':
  uniprot_to_taxonomy = {}
  targets_to_taxonomy = {}

  for crossref in tqdm(crossref_paths, desc='Reading Crossrefs'):
    path = os.path.join(dataset_path, crossref)
    df = pd.read_csv(path)
    df = df[['accessions','taxonomy_id']]
    df = df.drop_duplicates()
    df['accessions'] = df['accessions'].apply(ast.literal_eval)
    df['taxonomy_id'] = df['taxonomy_id'].apply(ast.literal_eval)
    df = df.explode('accessions')
    for _, row in df.iterrows():
      uniprot_to_taxonomy[row['accessions']] = row['taxonomy_id'][0]

  not_found = 0
  with open(targets_path, 'r') as fin:
    for line in tqdm(fin.readlines()[1:], desc='Mapping uniprot ids to taxonomy ids'):
      line = line.strip().split(',')
      targets = line[1]

      if targets == 'NOT_FOUND': continue

      targets_list = targets.split(' ')
      for target in targets_list:
        if target not in uniprot_to_taxonomy:
          target_taxonomy = uniprot_to_taxonomy_id(target)
          if target_taxonomy:
            uniprot_to_taxonomy[target] = target_taxonomy

        if target not in targets_to_taxonomy:
          if target in uniprot_to_taxonomy:
            targets_to_taxonomy[target] = uniprot_to_taxonomy[target]
          else:
            print(f'{target} not found in crossref')
            not_found += 1
  
  print(f'Mapped taxonomy of {len(targets_to_taxonomy)} targets | {not_found} not found in Crossrefs')

  with open(output_path, 'w') as fout:
    fout.write('uniprot_id,taxonomy_id\n')
    for target in tqdm(targets_to_taxonomy, desc=f'Saving mapping to file: {output_path}'):
      line = f'{target},{targets_to_taxonomy[target]}\n'
      fout.write(line)
  
  result = count_uniprot_by_taxonomy(output_path)
    
  if result is not None:
      print("Conteggio UniProt ID per Taxonomy ID:")
      print("-" * 40)
      print(result.to_string(index=False))
      
      # Statistiche riassuntive
      print(f"\nTotale taxonomy_id: {len(result)}")
      print(f"Totale uniprot_id: {result['uniprot_count'].sum()}")
      
      # Salva risultato in CSV
      output_file = output_path.replace('.csv', '_counts.csv')
      result.to_csv(output_file, index=False)
      print(f"Risultato salvato in: {output_file}")


