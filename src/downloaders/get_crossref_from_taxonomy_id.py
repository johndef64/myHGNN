import time
import json
import random
import asyncio
import aiohttp
import argparse
from typing import Dict, List, Set
from collections import defaultdict

class UniProtCrawler:
    def __init__(self, max_concurrent=10, delay_range=(0.1, 0.3)):
        self.max_concurrent = max_concurrent
        self.delay_range = delay_range
        self.session = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_sparql_query(self, taxonomy_id: str) -> str:
        return """
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

    async def fetch_uniprot_ids(self, taxonomy_id: str) -> List[str]:
        """Fetch UniProt IDs usando query SPARQL ottimizzata"""
        sparql_query = self.get_sparql_query(taxonomy_id)
        url = "https://sparql.uniprot.org/sparql"
        
        params = {"query": sparql_query, "format": "json"}
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                print(f"SPARQL query failed for taxonomy {taxonomy_id}: {response.status}")
                return []
                
            results = await response.json()
            
        protein_ids = []
        for binding in results["results"]["bindings"]:
            protein_uri = binding['protein']['value']
            protein_id = protein_uri.split('/')[-1]
            protein_ids.append(protein_id)
            
        return protein_ids

    async def get_cross_references_batch(self, uniprot_ids: List[str], target_dbs: Set[str]) -> List[Dict]:
        """Fetch cross-references in batch usando requests asincrone"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_single(uniprot_id: str) -> Dict:
            async with semaphore:
                await asyncio.sleep(random.uniform(*self.delay_range))
                return await self._fetch_cross_references(uniprot_id, target_dbs)
        
        tasks = [fetch_single(uid) for uid in uniprot_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtra errori e restituisce solo risultati validi
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error fetching {uniprot_ids[i]}: {result}")
            else:
                valid_results.append(result)
                
        return valid_results

    async def _fetch_cross_references(self, uniprot_id: str, target_dbs: Set[str]) -> Dict:
        """Fetch singolo cross-reference con retry logic"""
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        headers = {"User-Agent": random.choice(self.user_agents)}
        
        for attempt in range(3):  # Retry fino a 3 volte
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cross_references(uniprot_id, data, target_dbs)
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"HTTP {response.status} for {uniprot_id}")
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {uniprot_id}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    
        return {"UniProtID": uniprot_id, "error": "Failed after retries"}

    def _parse_cross_references(self, uniprot_id: str, data: Dict, target_dbs: Set[str]) -> Dict:
        """Parse cross-references in modo efficiente"""
        xrefs = data.get('uniProtKBCrossReferences', [])
        
        row = {"UniProtID": uniprot_id}
        db_dict = defaultdict(list)
        
        for xref in xrefs:
            db = xref.get("database")
            if db in target_dbs:
                db_dict[db].append(xref.get("id"))
        
        # Aggiungi tutte le DB anche se vuote per consistenza
        for db in target_dbs:
            row[db] = db_dict.get(db, [])
            
        return row

    def save_results(self, results: List[Dict], taxonomy_id: str):
        """Salva risultati con formato JSON valido"""
        output_file = f'{taxonomy_id}_uniprot_crossref.json'
        
        # Salva come array JSON valido invece di JSONL
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved {len(results)} entries to {output_file}")


async def main(taxonomy_id):
    target_dbs = {
        'EMBL', 'AlphaFoldDB', 'STRING', 'PaxDb', 'GeneID', 
        'KEGG', 'eggNOG', 'OrthoDB', 'GO', 'InterPro', 
        'PANTHER', 'Pfam'
    }
    
    async with UniProtCrawler(max_concurrent=15, delay_range=(0.05, 0.15)) as crawler:
        print(f"Processing taxonomy {taxonomy_id}...")
        
        # Fase 1: Ottieni IDs
        start_time = time.time()
        uniprot_ids = await crawler.fetch_uniprot_ids(taxonomy_id)
        print(f"Found {len(uniprot_ids)} proteins in {time.time() - start_time:.2f}s")
        
        if not uniprot_ids:
            print('ERR')
            return
            
        # Fase 2: Fetch cross-references in batch
        print("Fetching cross-references...")
        start_time = time.time()
        
        # Processa in chunks per evitare memory issues
        chunk_size = 500
        all_results = []
        
        for i in range(0, len(uniprot_ids), chunk_size):
            chunk = uniprot_ids[i:i+chunk_size]
            chunk_results = await crawler.get_cross_references_batch(chunk, target_dbs)
            all_results.extend(chunk_results)
            print(f"Processed {len(all_results)}/{len(uniprot_ids)} proteins")
        
        print(f"Completed in {time.time() - start_time:.2f}s")
        
        # Salva risultati
        crawler.save_results(all_results, taxonomy_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--taxonomy_id', '-t', required=True)
    args = parser.parse_args()
    taxonomy_id = args.taxonomy_id
    asyncio.run(main(taxonomy_id))