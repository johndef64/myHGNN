import requests
from xml.etree import ElementTree as ET


def get_total_time(start, end, precision=2):
  return round((end-start), precision)

def is_eukaryote(taxonomy_id):
  try:
    # Query NCBI Taxonomy API
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'taxonomy',
        'id': taxonomy_id,
        'retmode': 'xml'
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    
    lineage_elem = root.find('.//Lineage')
    if lineage_elem is None:
        raise ValueError(f"Lineage non trovata per taxonomy ID {taxonomy_id}")
    
    lineage = lineage_elem.text.lower()
    
    eukaryote_keywords = ['eukaryota', 'eukarya']
    
    return any(keyword in lineage for keyword in eukaryote_keywords)
      
  except requests.RequestException as e:
    raise RuntimeError(f"Errore nella richiesta NCBI: {e}")
  except ET.ParseError as e:
    raise RuntimeError(f"Errore parsing XML: {e}")