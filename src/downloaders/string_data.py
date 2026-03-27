#%%
import os
import pandas as pd
import glob

# filefolder = os.path.dirname(os.path.abspath(__file__))
if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')
os.getcwd()
#%%
os.chdir('dataset')
enseml_genes = pd.read_csv('ensembl_data.zip')
enseml_genes
#%%
datafolder = os.path.join('STRING')

os.chdir(datafolder)
# gallus gallus 9031 , canis lupus 9615
human = 9606
pathogens =[83332, 224308, 208964, 99287, 71421, 243230, 
            85962, 171101, 243277, 294, 1314, 272631,
            212717, 36329, 237561, 6183, 5664, 185431, 330879]

"""
9606,3369,Homo sapiens,
36329,100,Plasmodium falciparum 3D7,
83332,76,Mycobacterium tuberculosis H37Rv,
224308,66,Bacillus subtilis subsp. subtilis str. 168,
208964,46,Pseudomonas aeruginosa PAO1,
99287,33,Salmonella enterica subsp. enterica serovar Typhimurium str. LT2,
71421,21,Haemophilus influenzae Rd KW20,
243230,11,Deinococcus radiodurans R1 = ATCC 13939 = DSM 20539,
85962,10,Helicobacter pylori 26695,
237561,8,Candida albicans SC5314,
"""

# get filenames list with glob in working directory
pattern = '*.gz'
filenames = glob.glob(pattern)
print(filenames)
filenames = {
    # Interazioni funzionali
    1: '{}.protein.links{}.v12.0.txt.gz',          # Links funzionali 

    # Interazioni fisiche
    2: '{}.protein.physical.links{}.v12.0.txt.gz',          # Links fisici 
 
    # Annotazioni funzionali
    3: '{}.protein.enrichment.terms.v12.0.txt.gz', # Arricchimento funzionale
    
    # Relazioni evolutive (ortologia/omologia)
    4: '{}.protein.orthology.v12.0.txt.gz',      # Ortologhi
    #5: '{}.protein.homology.v12.0.txt.gz',       # Omologhi

    # Metadati e informazioni base
    6: '{}.protein.info.v12.0.txt.gz',           # Informazioni base proteine
    7: 'string_species.v12.0.txt',               # Lista specie
    8: 'species.tree.v12.0.txt',                 # Albero filogenetico
    9: '{}.protein.aliases.v12.0.txt.gz',              # Tassonomia
}

species_tree = pd.read_csv(filenames[8], sep='\t')
species_tree
pathogens_tree = species_tree[species_tree['#taxon_id'].isin(pathogens)]
pathogens_tree
# %%
##### PPIs #####
pathogen = pathogens[0]
details = ""   # "" for base, ".detailed" for detailed, ".full" for full

## LINKS ##
p_links = pd.read_csv(filenames[1].format(pathogen, details), sep=' ', compression='gzip')
# pd.read_csv(filenames[1].format(human, details), sep=' ', compression='gzip')
#%%
h_links = pd.read_csv(filenames[1].format(human, ""), sep=' ', compression='gzip')
import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.hist(h_links["combined_score"], bins=50, color='skyblue', edgecolor='black')
# plt.title("Distribution of Combined Scores in Human Physical Links")
# plt.xlabel("Combined Score")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()
h_links
#%%
h_links["normalized_combined_score"] = (h_links["combined_score"] - h_links["combined_score"].min()) / (h_links["combined_score"].max() - h_links["combined_score"].min())

h_links[h_links["normalized_combined_score"] > 0.6].sort_values(by="combined_score", ascending=False)#.head(10)
#%%



# %%
## PHYSICAL ##
p_string = pd.read_csv(filenames[2].format(pathogen, details), sep=' ', compression='gzip')
h_string = pd.read_csv(filenames[2].format(human, details), sep=' ', compression='gzip')
h_string

# %%
# Type of action STRNG v10
h_string_int = pd.read_csv("9606.protein.actions.v11.0.txt.gz", sep='\t', compression='gzip')
h_string_int.value_counts("mode")
#%%

h_linksv11=pd.read_csv("9606.protein.links.detailed.v11.0.txt.gz", sep=' ', compression='gzip')
h_linksv11
#%%
h_linksv11["normalized_combined_score"] = (h_linksv11["combined_score"] - h_linksv11["combined_score"].min()) / (h_linksv11["combined_score"].max() - h_linksv11["combined_score"].min())

h_linksv11["normalized_experimental"] = (h_linksv11["experimental"] - h_linksv11["experimental"].min()) / (h_linksv11["experimental"].max() - h_linksv11["experimental"].min())

#h_linksv11[h_linksv11["normalized_combined_score"] > 0.6].sort_values(by="combined_score", ascending=False)#.head(10)

h_linksv11["normalized_combined_score"] 
# h_linksv11["normalized_experimental"] 
#%%
h_linksv11["normalized_experimental"] 
# %%
## ANNOTATION (GO)##
ann = pd.read_table(filenames[3].format(pathogen), compression='gzip')

print(ann.category.drop_duplicates())
ann
# %%
## PROT INFO##
info = pd.read_table(filenames[6].format(pathogen), compression='gzip')
info
#%%
h_prot_info = pd.read_table(filenames[6].format("9606"), compression='gzip')
# h_prot_info["ensembl_gene_id"] = h_prot_info["#string_protein_id"].str.split('.').str[1]
# h_prot_info["ensembl_gene_id"].to_csv("human_ensembl_gene_ids.csv", index=False)
h_prot_info = pd.read_table("human_ensembl_gene_ids.csv", sep=',')
h_prot_info
#%%
##### PROT ALIASES #####
h_prot_alias = pd.read_table(filenames[9].format(pathogen), compression='gzip')
h_prot_alias

h_prot_alias_pivot = (h_prot_alias
                      .pivot_table(index='#string_protein_id', columns='source', values='alias', aggfunc='first')
                      .reset_index()
                      .rename(columns={'#string_protein_id': 'string_protein_id'}))
h_prot_alias_pivot.columns.name = None
#%%
h_prot_alias_pivot
#%%
##### Orthology #####
p_ortho = pd.read_table(filenames[4].format("330879"), compression='gzip')
p_ortho

#%%

def load_orthology_data(taxa_id, filter_og=None):
    filename = filenames[4].format(taxa_id)
    ortho = pd.read_csv(filename, sep='\t', compression='gzip')
    # merge with species tree
    ortho = pd.merge(ortho, species_tree, left_on='taxonomy_level', right_on='#taxon_id', how='left')
    ortho = ortho[['#protein', 'taxonomy_level', 'orthologous_group_or_ortholog',
            'taxon_name']]
    ortho["type"] =  ortho.orthologous_group_or_ortholog.str[0:3]
    if filter_og:
        ortho = ortho[ortho.type.isin([filter_og])]

    return ortho

OG = "COG"
h_ortho = load_orthology_data("9606", OG)
p_ortho = load_orthology_data(pathogens[0], OG)   #pathogens
h_ortho = h_ortho.rename(columns={'#protein': 'human_protein'}   )
p_ortho = p_ortho.rename(columns={'#protein': 'pathogen_protein'})
h_ortho_full = load_orthology_data("9606")
p_ortho_full = load_orthology_data(pathogens[0])
h_ortho
#%%
# h_ortho[h_ortho.orthologous_group_or_ortholog == "NOG019602"]
# p_ortho[p_ortho.orthologous_group_or_ortholog == "NOG019602"]
h_ortho
#%%
p_ortho.pathogen_protein.drop_duplicates().reset_index(), h_ortho.human_protein.drop_duplicates().reset_index()
# h_ortho_full["#protein"].drop_duplicates()


#%%
# Merge pathogen on human orthology data on human only common orthologs
merge_ortho = pd.merge(p_ortho, h_ortho, on='orthologous_group_or_ortholog', how='left')

merge_ortho.orthologous_group_or_ortholog.drop_duplicates()
# merge_ortho.pathogen_protein.drop_duplicates()
merge_ortho
# %%
h_ortho.orthologous_group_or_ortholog.value_counts()

# %%

merge_ortho = pd.merge(p_ortho_full, h_ortho_full, on='taxonomy_level', how='inner')

merge_ortho.orthologous_group_or_ortholog.drop_duplicates()
# merge_ortho.pathogen_protein.drop_duplicates()
merge_ortho
#%%
merge_ortho.taxonomy_level.value_counts()



# %%


# %%
#### calcoli

cog = pd.read_table(r"C:\Users\Utente\Downloads\COG.mappings.v11.0.txt.gz", compression='gzip')
#%%
cog[cog["##protein"].str.contains("224308")]



#%%
filename = filenames[4].format("9606")
h_ortho = pd.read_csv(filename, sep='\t', compression='gzip')

#%%

# Get GO annotations for a specific protein


uniprotid = "A0A067Z9B6"  # Example UniProt ID
import requests; print(requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprotid}?fields=go_id,go_p,go_c,go_f").json())
#%%

uniprotid = "A0A067Z9B6"
uniprotid = "O75964"  # Ensure UniProt ID is in the correct format
import requests
data = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprotid}?fields=go_id,go_p,go_c,go_f").json()
go_terms = []
for ref in data.get('uniProtKBCrossReferences', []):
    if ref.get('database') == 'GO':
        go_id = ref['id']
        for prop in ref.get('properties', []):
            if prop['key'] == 'GoTerm':
                go_term = prop['value']
                go_type = go_term.split(':')[0] if ':' in go_term else 'Unknown'
                go_terms.append({'id': go_id, 'term': go_term, 'type': go_type})

for term in go_terms:
    type_dict = {
        'P': 'BiologicalProcess',
        'C': 'CellularComponent',
        'F': 'MolecularFunction'
    }

    print(f"ID: {term['id']}, Type: {type_dict[term['type']]}")  #, Term: {term['term']}")
