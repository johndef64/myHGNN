import requests
from bs4 import BeautifulSoup
import time
import random
from tqdm.auto import tqdm
import re
import csv
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

DRUGBANK_URL = 'https://go.drugbank.com/bio_entities/'
OUTPUT_FILE = 'drugbank_to_uniprot_mapping.csv'
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
                writer.writerow(['DrugBank_ID', 'UniProt_ID'])
                for db_id, uni_id in mapping.items():
                    writer.writerow([db_id, uni_id])
            logger.info(f"Saved {len(mapping)} mappings to file")
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")

def extract_uniprot_id(html_content):
    """Extract UniProt ID from HTML"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Method 1: UniProtKB Entry field
        dt = soup.find('dt', id='uniprotkb-entry')
        if dt:
            dd = dt.find_next_sibling('dd')
            if dd:
                link = dd.find('a')
                if link and 'uniprot.org' in link.get('href', ''):
                    uniprot_id = link.get('href').split('/')[-1]
                    if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                        return uniprot_id
        
        # Method 2: External identifiers table
        external_table = soup.find('table', {'id': 'external-identifiers'})
        if external_table:
            rows = external_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2 and 'UniProtKB' in cells[0].get_text():
                    link = cells[1].find('a')
                    if link and 'uniprot.org' in link.get('href', ''):
                        uniprot_id = link.get('href').split('/')[-1]
                        if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                            return uniprot_id
        
        # Method 3: Any uniprot links
        uniprot_links = soup.find_all('a', href=lambda x: x and 'uniprot.org/uniprot/' in x)
        for link in uniprot_links:
            uniprot_id = link.get('href').split('/')[-1]
            if re.match(r'^[A-Z0-9]{6,10}$', uniprot_id):
                return uniprot_id
        
        return None
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

def scrape_drugbank_id(drugbank_id, session):
    """Scrape single DrugBank ID"""
    url = f"{DRUGBANK_URL}{drugbank_id}"
    
    for attempt in range(3):
        try:
            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            response = session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                uniprot_id = extract_uniprot_id(response.text)
                if uniprot_id:
                    found_counter.increment()
                    logger.info(f"Found: {drugbank_id} -> {uniprot_id}")
                    return drugbank_id, uniprot_id
                else:
                    not_found_counter.increment()
                    logger.warning(f"No UniProt ID found for: {drugbank_id}")
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
            for line in fin.readlines()[1:]:
                targets = line.strip().split(',')[1].split(' ')
                for target in targets:
                    if target.strip():
                        drugbank_ids_to_map.add(target.strip())
    except FileNotFoundError:
        logger.warning(f"File {path} not found, using test data")
        drugbank_ids_to_map = {'BE0000048'}
    
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