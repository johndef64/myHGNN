#!/usr/bin/env python3
"""
DrugBank Web Scraper
Estrae informazioni sui compound da DrugBank in modo robusto e multi-threaded
"""

import csv
import time
import random
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from typing import Dict, Optional, List, Tuple
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# set working director, two diurectory up if current is src
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drugbank_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DrugBankScraper:
    def __init__(self, max_workers: int = 5, delay_range: Tuple[float, float] = (1.0, 3.0)):
        self.max_workers = max_workers
        self.delay_range = delay_range
        self.base_url = "https://go.drugbank.com/drugs/"
        self.session_lock = threading.Lock()
        
        # Headers per sembrare un browser reale
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Inizializza le sessioni per ogni thread
        self.sessions = {}
        
    def create_session(self) -> requests.Session:
        session = requests.Session()
        
        # Configurazione retry
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        
        return session
    
    def get_session(self) -> requests.Session:
        thread_id = threading.current_thread().ident
        
        with self.session_lock:
            if thread_id not in self.sessions:
                self.sessions[thread_id] = self.create_session()
            return self.sessions[thread_id]
    
    def random_delay(self):
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def extract_external_ids(self, soup: BeautifulSoup) -> Dict[str, Optional[str]]:
        ids = {
            'pubchem': None,
            'chebi': None,
            'chembl': None
        }
        
        try:
            # Cerca nella sezione External Links
            external_links_section = soup.find('dt', string='External Links')
            if external_links_section:
                external_links_dd = external_links_section.find_next_sibling('dd')
                if external_links_dd:
                    # PubChem Compound
                    pubchem_link = external_links_dd.find('a', href=lambda x: x and 'pubchem.ncbi.nlm.nih.gov' in x and 'cid=' in x)
                    if pubchem_link:
                        href = pubchem_link.get('href', '')
                        if 'cid=' in href:
                            ids['pubchem'] = href.split('cid=')[-1].split('&')[0]
                    
                    # ChEBI
                    chebi_link = external_links_dd.find('a', href=lambda x: x and 'ebi.ac.uk/chebi' in x)
                    if chebi_link:
                        ids['chebi'] = chebi_link.text.strip()
                    
                    # ChEMBL
                    chembl_link = external_links_dd.find('a', href=lambda x: x and 'chembldb' in x or 'ebi.ac.uk/chembl' in x)
                    if chembl_link:
                        ids['chembl'] = chembl_link.text.strip()
            
            # Metodo alternativo: cerca in tutta la pagina
            if not any(ids.values()):
                # PubChem alternativo
                all_pubchem_links = soup.find_all('a', href=lambda x: x and 'pubchem.ncbi.nlm.nih.gov' in x and 'cid=' in x)
                for link in all_pubchem_links:
                    href = link.get('href', '')
                    if 'cid=' in href:
                        ids['pubchem'] = href.split('cid=')[-1].split('&')[0]
                        break
                
                # ChEBI alternativo
                all_chebi_links = soup.find_all('a', href=lambda x: x and 'ebi.ac.uk/chebi' in x)
                for link in all_chebi_links:
                    text = link.text.strip()
                    if text.isdigit():
                        ids['chebi'] = text
                        break
                
                # ChEMBL alternativo
                all_chembl_links = soup.find_all('a', href=lambda x: x and ('chembldb' in x or 'ebi.ac.uk/chembl' in x))
                for link in all_chembl_links:
                    text = link.text.strip()
                    if text.startswith('CHEMBL'):
                        ids['chembl'] = text
                        break
                        
        except Exception as e:
            logger.warning(f"Errore nell'estrazione degli ID: {e}")
        
        return ids
    
    def scrape_drug_info(self, drug_id: str) -> Dict[str, Optional[str]]:
        result = {
            'drugbank_id': drug_id,
            'pubchem_id': None,
            'chebi_id': None,
            'chembl_id': None,
            'status': 'failed'
        }
        
        try:
            session = self.get_session()
            url = f"{self.base_url}{drug_id}"
            
            logger.info(f"Scraping {drug_id} da {url}")
            
            # Applica delay casuale
            self.random_delay()
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Estrai gli ID esterni
            external_ids = self.extract_external_ids(soup)
            
            result.update({
                'pubchem_id': external_ids['pubchem'],
                'chebi_id': external_ids['chebi'],
                'chembl_id': external_ids['chembl'],
                'status': 'success'
            })
            
            logger.info(f"Completato {drug_id}: PubChem={external_ids['pubchem']}, ChEBI={external_ids['chebi']}, ChEMBL={external_ids['chembl']}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore di rete per {drug_id}: {e}")
            result['status'] = 'network_error'
        except Exception as e:
            logger.error(f"Errore generico per {drug_id}: {e}")
            result['status'] = 'error'
        
        return result
    
    def read_drug_ids(self, csv_file: str) -> List[str]:
        """
        Legge gli ID dei farmaci dal file CSV
        
        Args:
            csv_file: Path del file CSV
            
        Returns:
            Lista degli ID unici
        """
        drug_ids = set()
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Salta l'header
                
                for row in reader:
                    if row and len(row) > 0:
                        drug_id = row[0].strip()
                        if drug_id:
                            drug_ids.add(drug_id)
            
            logger.info(f"Letti {len(drug_ids)} ID unici dal file {csv_file}")
            return list(drug_ids)
            
        except Exception as e:
            logger.error(f"Errore nella lettura del file {csv_file}: {e}")
            return []
    
    def save_results(self, results: List[Dict], output_file: str):
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['drugbank_id', 'pubchem_id', 'chebi_id', 'chembl_id', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            
            logger.info(f"Risultati salvati in {output_file}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio: {e}")
    
    def save_checkpoint(self, results: List[Dict], checkpoint_file: str):
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Checkpoint salvato in {checkpoint_file}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio del checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> List[Dict]:
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Checkpoint caricato da {checkpoint_file}")
                return results
        except Exception as e:
            logger.error(f"Errore nel caricamento del checkpoint: {e}")
        return []
    
    def run(self, input_csv: str, output_csv: str, checkpoint_file: str = 'checkpoint.json'):
        existing_results = self.load_checkpoint(checkpoint_file)
        processed_ids = {result['drugbank_id'] for result in existing_results}
        
        drug_ids = self.read_drug_ids(input_csv)
        
        remaining_ids = [drug_id for drug_id in drug_ids if drug_id not in processed_ids]
        
        logger.info(f"ID totali: {len(drug_ids)}, già processati: {len(processed_ids)}, rimanenti: {len(remaining_ids)}")
        
        if not remaining_ids:
            logger.info("Tutti gli ID sono già stati processati")
            self.save_results(existing_results, output_csv)
            return
        
        results = existing_results.copy()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.scrape_drug_info, drug_id): drug_id 
                for drug_id in remaining_ids
            }
            
            for i, future in enumerate(as_completed(future_to_id)):
                drug_id = future_to_id[future]
                
                try:
                    result = future.result()
                    results.append(result)
                  
                    if (i + 1) % 10 == 0:
                        self.save_checkpoint(results, checkpoint_file)
                        logger.info(f"Progresso: {i + 1}/{len(remaining_ids)} completati")
                    
                except Exception as e:
                    logger.error(f"Errore nel processare {drug_id}: {e}")
                    results.append({
                        'drugbank_id': drug_id,
                        'pubchem_id': None,
                        'chebi_id': None,
                        'chembl_id': None,
                        'status': 'exception'
                    })
        
        self.save_results(results, output_csv)
        self.save_checkpoint(results, checkpoint_file)
        
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Scraping completato: {successful}/{len(results)} successi")

def main():
    INPUT_CSV = 'dataset/drug_target_uniprotid.csv'
    OUTPUT_CSV = 'drugbank_mapping.csv'
    CHECKPOINT_FILE = 'scraping_checkpoint.json'
    MAX_WORKERS = 6 
    DELAY_RANGE = (0.8, 1.5) 
    
    if not os.path.exists(INPUT_CSV):
        logger.error(f"File di input {INPUT_CSV} non trovato!")
        return
      
    scraper = DrugBankScraper(max_workers=MAX_WORKERS, delay_range=DELAY_RANGE)
    
    try:
        scraper.run(INPUT_CSV, OUTPUT_CSV, CHECKPOINT_FILE)
        logger.info("Scraping completato con successo!")    
    except KeyboardInterrupt:
        logger.info("Scraping interrotto dall'utente")
    except Exception as e:
        logger.error(f"Errore durante il scraping: {e}")

if __name__ == "__main__":
    main()