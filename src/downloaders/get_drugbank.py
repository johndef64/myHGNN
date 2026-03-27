
#%%
import pandas as pd
import gzip
from Bio import SwissProt
import os
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
os.chdir("dataset")

root = "D:\Data\DrugBank"

# df = pd.read_csv(f"{root}/drugbank.csv.gz",  low_memory=False)
# df
#%%
df = pd.read_csv(f"{root}/drugbank_clean.csv",  low_memory=False)
df

# %%
df[df["drugbank-id"] == "BE0000048"]  # Aspirin
df[df["targets"] == "BE0000048"]  
# %%
df.columns

cols = ['drugbank-id','targets', 'carriers', 'pathways', 'reactions']
df[cols].drop_duplicates().reset_index(drop=True)#.head(10)

# %%

drugmapping = df[cols].drop_duplicates().reset_index(drop=True)
drugmapping.to_csv(f"drugbank_mapping.csv.gz", index=False, compression='gzip')
#%%
drugmapping

#%%
###########################
# PharmKG

file = r"D:\Data\PharmKG\raw_PharmKG-180k.zip"

pahrmkg_df = pd.read_csv(file, compression='zip', low_memory=False)
pahrmkg_df