import requests
import json
import time

# Lista di ChEMBL ID da convertire
CHEMBL_IDS = [
    "CHEMBL2074",
    # Aggiungi altri ID qui...
]

def chembl_to_pubchem(chembl_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chembl_id}/cids/JSON"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
        else:
            print(f"[WARN] {chembl_id}: status {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] {chembl_id}: {e}")
        return None

def main():
    result = {}
    for chembl_id in CHEMBL_IDS:
        print(f"Processing {chembl_id}...")
        pubchem_cid = chembl_to_pubchem(chembl_id)
        result[chembl_id] = pubchem_cid
        print(f"  -> PubChem CID: {pubchem_cid}")
        time.sleep(0.3)

    with open("chembl_to_pubchem.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSalvato in chembl_to_pubchem.json")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()