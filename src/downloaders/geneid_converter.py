import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional

class EnsemblConverter:
    def __init__(self):
        self.session = None
        self.base_url = "https://rest.ensembl.org"
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def _fetch(self, url: str, params: dict = None, retries: int = 3) -> dict:
        """Fetch con retry robusto"""
        for attempt in range(retries):
            try:
                await asyncio.sleep(0.1 * attempt)  # Backoff progressivo
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:  # Rate limit
                        await asyncio.sleep(2)
                        continue
                    else:
                        print(f"HTTP {resp.status} for {url}")
                        return {}
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt == retries - 1:
                    return {}
                await asyncio.sleep(1)
        return {}
    
    async def convert_ensembl_id(self, ensembl_id: str) -> Dict:
        """Converte Ensembl ID con logica sequenziale per evitare race conditions"""
        
        if ensembl_id.startswith('ENSG'):
            id_type = 'gene'
        elif ensembl_id.startswith('ENSP'):
            id_type = 'protein'
        else:
            return {'error': f'ID non riconosciuto: {ensembl_id}'}
        
        result = {
            'input_id': ensembl_id,
            'input_type': id_type,
            'ensembl_gene_id': None,
            'ensembl_protein_ids': [],
            'uniprot_ids': [],
            'ncbi_gene_id': None,
            'gene_symbol': None,
            'gene_name': None,
            'chromosome': None,
            'strand': None
        }
        
        try:
            if id_type == 'gene':
                # Per ENSG: prima le info del gene, poi xrefs, poi proteine
                gene_info = await self._get_gene_info(ensembl_id)
                if gene_info:
                    result['ensembl_gene_id'] = ensembl_id
                    result['gene_symbol'] = gene_info.get('display_name')
                    result['gene_name'] = gene_info.get('description')
                    result['chromosome'] = gene_info.get('seq_region_name')
                    result['strand'] = gene_info.get('strand')
                
                # Xrefs per il gene
                xrefs = await self._get_xrefs(ensembl_id)
                await self._parse_xrefs(xrefs, result)
                
                # Proteine del gene
                protein_ids = await self._get_protein_ids(ensembl_id)
                result['ensembl_protein_ids'] = protein_ids
                
            else:  # ENSP
                # Per ENSP: prima info proteina, poi trova il gene, poi completa
                protein_info = await self._get_gene_info(ensembl_id)
                result['ensembl_protein_ids'] = [ensembl_id]
                
                if protein_info:
                    # Se ha Parent, è il gene ID
                    parent_id = protein_info.get('Parent')
                    if parent_id:
                        result['ensembl_gene_id'] = parent_id
                        
                        # Ottieni info del gene parent
                        gene_info = await self._get_gene_info(parent_id)
                        if gene_info:
                            result['gene_symbol'] = gene_info.get('display_name')
                            result['gene_name'] = gene_info.get('description')
                            result['chromosome'] = gene_info.get('seq_region_name')
                            result['strand'] = gene_info.get('strand')
                
                # Xrefs per la proteina
                xrefs = await self._get_xrefs(ensembl_id)
                await self._parse_xrefs(xrefs, result)
                
                # Se abbiamo trovato il gene, prendi anche le sue xrefs
                if result['ensembl_gene_id']:
                    gene_xrefs = await self._get_xrefs(result['ensembl_gene_id'])
                    await self._parse_xrefs(gene_xrefs, result)
        
        except Exception as e:
            print(f"Error processing {ensembl_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _parse_xrefs(self, xrefs: list, result: dict):
        """Parse cross-references e aggiorna result"""
        if not isinstance(xrefs, list):
            return
            
        for xref in xrefs:
            db_name = xref.get('dbname', '')
            primary_id = xref.get('primary_id', '')
            
            if not primary_id:
                continue
                
            if db_name == 'EntrezGene':
                if not result['ncbi_gene_id']:  # Solo se non già settato
                    result['ncbi_gene_id'] = primary_id
                    
            elif db_name in ['Uniprot_gn', 'Uniprot/SWISSPROT', 'UniProtKB/Swiss-Prot', 'Uniprot/SPTREMBL']:
                if primary_id not in result['uniprot_ids']:
                    result['uniprot_ids'].append(primary_id)
                    
            elif db_name in ['HGNC_symbol', 'HGNC']:
                if not result['gene_symbol']:  # Solo se non già settato
                    result['gene_symbol'] = primary_id
    
    async def _get_xrefs(self, ensembl_id: str) -> list:
        """Get cross-references"""
        url = f"{self.base_url}/xrefs/id/{ensembl_id}"
        params = {'content-type': 'application/json'}
        result = await self._fetch(url, params)
        return result if isinstance(result, list) else []
    
    async def _get_gene_info(self, ensembl_id: str) -> dict:
        """Get informazioni gene/proteina"""
        url = f"{self.base_url}/lookup/id/{ensembl_id}"
        params = {'content-type': 'application/json'}
        return await self._fetch(url, params)
    
    async def _get_protein_ids(self, gene_id: str) -> List[str]:
        """Ottieni protein IDs da gene ID con expand"""
        url = f"{self.base_url}/lookup/id/{gene_id}"
        params = {'content-type': 'application/json', 'expand': '1'}
        result = await self._fetch(url, params)
        
        protein_ids = []
        if result and 'Transcript' in result:
            for transcript in result['Transcript']:
                if 'Translation' in transcript and transcript['Translation']:
                    protein_id = transcript['Translation'].get('id')
                    if protein_id and protein_id.startswith('ENSP'):
                        protein_ids.append(protein_id)
        
        return protein_ids
    
    async def convert_batch(self, ensembl_ids: List[str], batch_size: int = 8) -> pd.DataFrame:
        """Converte batch con rate limiting più conservativo"""
        results = []
        
        for i in range(0, len(ensembl_ids), batch_size):
            batch = ensembl_ids[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(ensembl_ids)-1)//batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} IDs)")
            
            # Processa batch in parallelo ma con semaforo per limitare concorrenza
            semaphore = asyncio.Semaphore(4)  # Max 4 chiamate simultanee
            
            async def limited_convert(eid):
                async with semaphore:
                    return await self.convert_ensembl_id(eid)
            
            batch_results = await asyncio.gather(
                *[limited_convert(eid) for eid in batch],
                return_exceptions=True
            )
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error for {batch[j]}: {result}")
                elif 'error' not in result:
                    results.append(result)
                else:
                    print(f"Failed: {result}")
            
            # Pausa più lunga tra batch
            if batch_num < total_batches:
                print(f"Waiting before next batch...")
                await asyncio.sleep(1.5)
        
        # Convert to DataFrame
        df_data = []
        for r in results:
            df_data.append({
                'Input_ID': r['input_id'],
                'Input_Type': r['input_type'],
                'Ensembl_Gene_ID': r['ensembl_gene_id'],
                'Ensembl_Protein_IDs': '; '.join(r['ensembl_protein_ids']) if r['ensembl_protein_ids'] else None,
                'UniProt_IDs': '; '.join(r['uniprot_ids']) if r['uniprot_ids'] else None,
                'NCBI_Gene_ID': r['ncbi_gene_id'],
                'Gene_Symbol': r['gene_symbol'],
                'Gene_Name': r['gene_name'],
                'Chromosome': r['chromosome'],
                'Strand': r['strand']
            })
        
        return pd.DataFrame(df_data)

# Funzioni di utilità (invariate)
async def convert_ensembl_ids(ensembl_ids: List[str], batch_size: int = 8) -> pd.DataFrame:
    async with EnsemblConverter() as converter:
        return await converter.convert_batch(ensembl_ids, batch_size)

def convert_ensembl_sync(ensembl_ids: List[str], batch_size: int = 8) -> pd.DataFrame:
    return asyncio.run(convert_ensembl_ids(ensembl_ids, batch_size))

async def convert_from_file(file_path: str, id_column: str, batch_size: int = 8) -> pd.DataFrame:
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    ensembl_ids = df[id_column].dropna().unique().tolist()
    
    conversion_table = await convert_ensembl_ids(ensembl_ids, batch_size)
    return df.merge(conversion_table, left_on=id_column, right_on='Input_ID', how='left')

# Esempio con i tuoi ID problematici
async def esempio():
    ensembl_ids = [
        'ENSP00000000233',  # Quello che dava problemi
        'ENSP00000000412',
        'ENSP00000001008',
        'ENSP00000001146'
    ]
    ensembl_ids = pd.read_csv('human_ensembl_gene_ids.csv')['ensembl_gene_id'].tolist()#[:10]  
    print("Converting Ensembl IDs...")
    df = await convert_ensembl_ids(ensembl_ids)
    print(df.to_string(index=False))
    
    df.to_csv('ensembl_conversions.csv', index=False)
    print("\nSaved to ensembl_conversions.csv")
    
    return df

if __name__ == "__main__":
    asyncio.run(esempio())