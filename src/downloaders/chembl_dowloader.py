#%%
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings
import pandas as pd
from tqdm import tqdm
import time
import os

Settings.Instance().TIMEOUT = 60

molecule = new_client.molecule
activity = new_client.activity
target = new_client.target

CHECKPOINT_FILE = "checkpoint_activities.csv"
TARGETS_FILE = "checkpoint_targets.csv"

# ── 1. Molecole approvate ────────────────────────────────────────────────────
print("Recupero molecole...")
approved_mols = molecule.filter(max_phase__gte=1).only(['molecule_chembl_id', 'pref_name', 'max_phase'])
approved_df = pd.DataFrame(approved_mols)
approved_ids = set(approved_df['molecule_chembl_id'].tolist())
print(f"Molecole: {len(approved_df)}")

# ── 2. Target batterici (con checkpoint) ────────────────────────────────────
if os.path.exists(TARGETS_FILE):
    print("Carico target da checkpoint...")
    target_df = pd.read_csv(TARGETS_FILE)
else:
    print("Recupero target batterici proteici...")
    bacterial_targets = target.filter(
        organism_taxonomy__l2="Bacteria",
        target_type="SINGLE PROTEIN"
    ).only(['target_chembl_id', 'pref_name', 'organism'])
    target_df = pd.DataFrame(bacterial_targets)
    target_df.to_csv(TARGETS_FILE, index=False)
    print(f"Salvati {len(target_df)} target in {TARGETS_FILE}")

target_ids = target_df['target_chembl_id'].tolist()
print(f"Target batterici proteici: {len(target_ids)}")

# ── 3. Carica checkpoint attività ────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
    all_activities = checkpoint_df.to_dict('records')
    done_targets = set(checkpoint_df['target_chembl_id'].dropna().unique())
    print(f"✅ Ripreso da checkpoint: {len(done_targets)}/{len(target_ids)} target già processati ({len(all_activities)} attività)")
else:
    all_activities = []
    done_targets = set()
    print("Nessun checkpoint trovato, parto da zero.")

remaining = [t for t in target_ids if t not in done_targets]
print(f"Target rimanenti: {len(remaining)}")

# ── 4. Fetch attività ────────────────────────────────────────────────────────
batch_size = 10
save_every = 20  # salva ogni 20 batch (~200 target)

for i in tqdm(range(0, len(remaining), batch_size), desc="Fetching"):
    batch = remaining[i:i+batch_size]
    
    for attempt in range(5):
        try:
            acts = activity.filter(
                target_chembl_id__in=batch,
                assay_type__in=["B", "F"]
            ).only([
                'molecule_chembl_id', 'target_chembl_id', 'target_pref_name',
                'target_organism', 'standard_type', 'standard_value',
                'standard_units', 'pchembl_value', 'assay_type'
            ])
            
            batch_acts = [a for a in acts if a['molecule_chembl_id'] in approved_ids]
            all_activities.extend(batch_acts)
            break

        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"\n  Tentativo {attempt+1}/5 fallito: {e} — aspetto {wait}s")
            time.sleep(wait)
    
    # Salva checkpoint
    if (i // batch_size + 1) % save_every == 0:
        pd.DataFrame(all_activities).to_csv(CHECKPOINT_FILE, index=False)
        tqdm.write(f"  💾 Checkpoint: {len(done_targets) + i + batch_size} target processati, {len(all_activities)} attività")
    
    time.sleep(1)

# Salva checkpoint finale
pd.DataFrame(all_activities).to_csv(CHECKPOINT_FILE, index=False)
print(f"\n💾 Checkpoint finale salvato.")

# ── 5. Risultato finale ──────────────────────────────────────────────────────
final_df = pd.DataFrame(all_activities).merge(
    approved_df[['molecule_chembl_id', 'pref_name', 'max_phase']],
    on='molecule_chembl_id', how='left'
)

final_df['drug_status'] = final_df['max_phase'].apply(
    lambda p: "Approved Drug" if p == 4 else "Clinical Candidate"
)

final_df.to_csv("bacterial_approved_clinical.csv", index=False)
print(f"\n✅ Righe totali: {len(final_df)}")
print(f"✅ Drug unici:   {final_df['molecule_chembl_id'].nunique()}")
print(f"✅ Target unici: {final_df['target_chembl_id'].nunique()}")
print(f"✅ Approved Drugs:      {(final_df['drug_status']=='Approved Drug').sum()}")
print(f"✅ Clinical Candidates: {(final_df['drug_status']=='Clinical Candidate').sum()}")