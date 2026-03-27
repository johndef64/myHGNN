import re
import os
import csv
import time
import random
import logging
import requests
import threading
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drugbank_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DRUGBANK_URL = 'https://go.drugbank.com/drugs/'
OUTPUT_FILE = 'drugbank_to_uniprot_targets_mapping.csv'
CHECKPOINT_INTERVAL = 20
MAX_WORKERS = 5

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
]

# Thread-safe counters
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

found_counter = ThreadSafeCounter()
not_found_counter = ThreadSafeCounter()
mapping_lock = threading.Lock()

def create_session():
    """Create session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_existing_mapping():
    """Load already scraped mappings"""
    mapping = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        mapping[row[0]] = row[1]
            logger.info(f"Loaded {len(mapping)} existing mappings")
        except Exception as e:
            logger.error(f"Error loading existing mappings: {e}")
    return mapping

def save_mapping(mapping):
    """Thread-safe save mapping"""
    with mapping_lock:
        try:
            with open(OUTPUT_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['DrugBank_ID', 'UniProt_IDs'])
                for db_id, uni_ids in mapping.items():
                    writer.writerow([db_id, uni_ids])
            logger.info(f"Saved {len(mapping)} mappings to file")
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")

def extract_uniprot_ids(html_content):
    """Extract UniProt IDs from targets section"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        uniprot_ids = []
        
        # Find targets section
        targets_section = soup.find('div', {'id': 'targets'})
        if targets_section:
            # Find all uniprot.org links in targets section
            uniprot_links = targets_section.find_all('a', href=lambda x: x and 'uniprot.org/uniprot/' in x)
            for link in uniprot_links:
                href = link.get('href')
                uniprot_id = href.split('/')[-1]
                if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                    uniprot_ids.append(uniprot_id)
        
        # Also check in mechanism of action table
        moa_table = soup.find('table', {'id': 'drug-moa-target-table'})
        if moa_table:
            uniprot_links = moa_table.find_all('a', href=lambda x: x and 'uniprot.org/uniprot/' in x)
            for link in uniprot_links:
                href = link.get('href')
                uniprot_id = href.split('/')[-1]
                if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                    uniprot_ids.append(uniprot_id)
        
        # Also check individual target cards
        target_cards = soup.find_all('div', class_='bond card')
        for card in target_cards:
            uniprot_links = card.find_all('a', href=lambda x: x and 'uniprot.org/uniprot/' in x)
            for link in uniprot_links:
                href = link.get('href')
                uniprot_id = href.split('/')[-1]
                if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                    uniprot_ids.append(uniprot_id)
        
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(uniprot_ids))
        return unique_ids if unique_ids else None
        
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

def scrape_drugbank_id(drugbank_id, session):
    """Scrape single DrugBank ID for UniProt targets"""
    url = f"{DRUGBANK_URL}{drugbank_id}"
    
    for attempt in range(3):
        try:
            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            response = session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                uniprot_ids = extract_uniprot_ids(response.text)
                if uniprot_ids:
                    found_counter.increment()
                    uniprot_str = ' '.join(uniprot_ids)
                    logger.info(f"Found: {drugbank_id} -> {uniprot_str}")
                    return drugbank_id, uniprot_str
                else:
                    not_found_counter.increment()
                    logger.warning(f"No UniProt IDs found for: {drugbank_id}")
                    return drugbank_id, 'NOT_FOUND'
            
            elif response.status_code in [429, 403, 503]:
                wait_time = (2 ** attempt) + random.uniform(1, 3)
                logger.warning(f"Rate limited for {drugbank_id}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP {response.status_code} for {drugbank_id}")
                break
                
        except Exception as e:
            logger.error(f"Error scraping {drugbank_id} (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    
    not_found_counter.increment()
    logger.error(f"Failed to scrape: {drugbank_id}")
    return drugbank_id, 'FAILED'

def main():
    # Load DrugBank IDs
    path = 'dataset/drugbank_mapping.csv'
    drugbank_ids_to_map = set()
    
    try:
        with open(path, 'r') as fin:
            reader = csv.reader(fin)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 0:
                    drugbank_id = row[0].strip()
                    if drugbank_id:
                        drugbank_ids_to_map.add(drugbank_id)
    except FileNotFoundError:
        logger.warning(f"File {path} not found, using test data")
        drugbank_ids_to_map = {'DB15796'}
    
    # Load existing mappings
    mapping = load_existing_mapping()
    already_processed = set(mapping.keys())
    remaining_ids = list(drugbank_ids_to_map - already_processed)
    
    logger.info(f"Total: {len(drugbank_ids_to_map)}, Processed: {len(already_processed)}, Remaining: {len(remaining_ids)}")
    
    if not remaining_ids:
        logger.info('All IDs already processed!')
        return
    
    # Initialize counters from existing data
    existing_found = len([v for v in mapping.values() if v not in ['NOT_FOUND', 'FAILED']])
    existing_not_found = len(mapping) - existing_found
    found_counter._value = existing_found
    not_found_counter._value = existing_not_found
    
    # Shuffle for better load distribution
    random.shuffle(remaining_ids)
    
    # Process with thread pool
    processed = 0
    pbar = tqdm(total=len(remaining_ids), 
                desc=f"Found: {found_counter.value}, Not found: {not_found_counter.value}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create sessions for each worker
        sessions = [create_session() for _ in range(MAX_WORKERS)]
        session_iter = iter(sessions * (len(remaining_ids) // MAX_WORKERS + 1))
        
        # Submit jobs
        future_to_id = {
            executor.submit(scrape_drugbank_id, drugbank_id, next(session_iter)): drugbank_id 
            for drugbank_id in remaining_ids
        }
        
        for future in as_completed(future_to_id):
            drugbank_id, result = future.result()
            mapping[drugbank_id] = result
            processed += 1
            
            # Update progress
            pbar.update(1)
            pbar.set_description(f"Found: {found_counter.value}, Not found: {not_found_counter.value}")
            
            # Checkpoint save
            if processed % CHECKPOINT_INTERVAL == 0:
                save_mapping(mapping)
                logger.info(f"Checkpoint: {processed}/{len(remaining_ids)} processed")
            
            # Rate limiting between batches
            time.sleep(random.uniform(0.1, 0.3))
    
    pbar.close()
    
    # Final save and statistics
    save_mapping(mapping)
    
    total_found = len([v for v in mapping.values() if v not in ['NOT_FOUND', 'FAILED']])
    total_not_found = len([v for v in mapping.values() if v == 'NOT_FOUND'])
    total_failed = len([v for v in mapping.values() if v == 'FAILED'])
    
    logger.info(f"Final results: {total_found} found, {total_not_found} not found, {total_failed} failed")
    logger.info(f"Success rate: {total_found/len(mapping)*100:.1f}%")

if __name__ == '__main__':
    main()