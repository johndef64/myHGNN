import json
import os
import csv
import time
import torch
import random
import numpy as np
import pandas as pd
from typing import *
from termcolor import colored
from collections import defaultdict
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

PARAMS_FILE = 'configurations/vitagraph.yml'

def set_seed(seed=42):
	np.random.seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

# Utility Functions to exrtact the LCC from the DRKG
class UnionFind:
	def __init__(self):
		self.parent = dict()
		self.rank = defaultdict(int)  # Optimization with rank compression

	def find(self, x):
		if x not in self.parent:
			self.parent[x] = x
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]

	def union(self, x, y):
		root_x = self.find(x)
		root_y = self.find(y)
		
		if root_x == root_y:
			return
			
		# Union by rank to optimize operations
		if self.rank[root_x] < self.rank[root_y]:
			self.parent[root_x] = root_y
		elif self.rank[root_x] > self.rank[root_y]:
			self.parent[root_y] = root_x
		else:
			self.parent[root_y] = root_x
			self.rank[root_x] += 1

def debug_print(message, debug=True):
	"""Utility function to print messages only if debug is active"""
	if debug:
		print(message)

def count_connected_components(tsv_file, output_file="output", debug=True):
	"""
	Analyzes the connected components in a knowledge graph in TSV format.
	
	Args:
		tsv_file: Path to the TSV file with 5 columns
		output_file: Output file
		debug: If True, prints debug messages
	
	Returns:
		Dict with statistics on the connected components
	"""
	debug_print(f"Analyzing file: {tsv_file}", debug)
	start_time = time.time()
	
	uf = UnionFind()
	edges_count = 0
	
	# First pass: build connected components
	debug_print("Building connected components...", debug)
	with open(tsv_file, 'r', encoding='utf-8') as f:
		tsv_reader = csv.reader(f, delimiter='\t')
		header = next(tsv_reader, None)  # Save header
		
		for i, row in enumerate(tsv_reader):
			if debug and i % 1000000 == 0 and i > 0:
				debug_print(f"  Processed {i:,} rows...", debug)
			try:
				if len(row) < 5:
					debug_print(f"Warning: row {i+2} does not have 5 columns: {row}", debug)
					continue
				head, _, tail, _, _ = row
				uf.union(head, tail)
				edges_count += 1
			except ValueError:
				debug_print(f"Warning: error at row {i+2}: {row}", debug)
				continue
	
	debug_print("Computing component statistics...", debug)
	
	# Collect component statistics
	component_sizes = defaultdict(int)
	node_to_component = {}
	
	for node in uf.parent:
		root = uf.find(node)
		component_sizes[root] += 1
		node_to_component[node] = root
	
	total_nodes = len(uf.parent)
	total_from_components = sum(component_sizes.values())
	num_components = len(component_sizes)
	
	# Verify data consistency
	assert total_nodes == total_from_components, "Error: inconsistent count!"
	
	# Sort components by size
	components_by_size = sorted(component_sizes.items(), key=lambda x: x[1], reverse=True)
	sizes = [size for _, size in components_by_size]
	
	# Identify the largest connected component (LCC)
	largest_component_id = components_by_size[0][0] if components_by_size else None
	largest_component_size = sizes[0] if sizes else 0
	
	# Extract LCC if requested
	if largest_component_id is not None:
		debug_print(f"Extracting LCC to file: {output_file}", debug)
		
		lcc_edges_count = 0
		lcc_nodes = set()

		# Check if the input and output files are the same
		same_file = os.path.abspath(tsv_file) == os.path.abspath(output_file)
		temp_output_file = output_file + ".temp" if same_file else output_file
		
		with open(tsv_file, 'r', encoding='utf-8') as fin, \
			 open(temp_output_file, 'w', encoding='utf-8', newline='') as fout:
			
			tsv_reader = csv.reader(fin, delimiter='\t')
			tsv_writer = csv.writer(fout, delimiter='\t')
			
			# Write header
			header = next(tsv_reader, None)
			if header:
				tsv_writer.writerow(header)
			
			# Second pass: extract LCC rows
			for i, row in enumerate(tsv_reader):
				if debug and i % 1000000 == 0 and i > 0:
					debug_print(f"  Filtered {i:,} rows for LCC...", debug)
				
				try:
					if len(row) < 5:
						continue
					
					head, _, tail, _, _ = row
					
					# Check if both nodes belong to the LCC
					head_component = node_to_component.get(head)
					tail_component = node_to_component.get(tail)
					
					if head_component == largest_component_id and tail_component == largest_component_id:
						tsv_writer.writerow(row)
						lcc_edges_count += 1
						lcc_nodes.add(head)
						lcc_nodes.add(tail)
						
				except Exception as e:
					debug_print(f"Error during LCC extraction at row {i+2}: {e}", debug)
					continue
		
		# If we used a temporary file, replace the original with it
		if same_file:
			debug_print(f"Input and output files are the same. Replacing original with filtered data.", debug)
			os.replace(temp_output_file, output_file)
			
		debug_print(f"LCC extracted: {len(lcc_nodes):,} nodes, {lcc_edges_count:,} edges", debug)
	
	# STATISTICS
	if debug:
		debug_print(f"\n===== BASIC STATISTICS =====", debug)
		debug_print(f"Number of connected components: {num_components:,}", debug)
		debug_print(f"Total number of unique nodes: {total_nodes:,}", debug)
		debug_print(f"Total number of edges: {edges_count:,}", debug)
		debug_print(f"Sum of component sizes: {total_from_components:,}", debug)
		
		debug_print(f"\n===== ADVANCED STATISTICS =====", debug)
		debug_print(f"Average component size: {np.mean(sizes):.2f} nodes", debug)
		debug_print(f"Median component size: {np.median(sizes):.2f} nodes", debug)
		debug_print(f"Standard deviation: {np.std(sizes):.2f}", debug)
		
		debug_print("\n===== TOP 10 COMPONENTS BY SIZE =====", debug)
		for i, (_, size) in enumerate(components_by_size[:10], 1):
			percent = (size / total_nodes) * 100
			debug_print(f"#{i}: {size:,} nodes ({percent:.2f}% of the graph)", debug)
		
		debug_print("\n===== COMPONENT SIZE DISTRIBUTION =====", debug)
		size_distribution = defaultdict(int)
		for size in sizes:
			size_distribution[size] += 1
		
		# Function to group sizes into brackets
		def get_size_bracket(size):
			if size == 1:
				return "1 (isolated)"
			elif size == 2:
				return "2 (pairs)"
			elif size <= 5:
				return "3-5"
			elif size <= 10:
				return "6-10"
			elif size <= 100:
				return "11-100"
			elif size <= 1000:
				return "101-1000"
			elif size <= 10000:
				return "1001-10000"
			else:
				return ">10000"
		
		brackets = defaultdict(int)
		for size, count in size_distribution.items():
			brackets[get_size_bracket(size)] += count
		
		# Predefined bracket order
		bracket_order = ["1 (isolated)", "2 (pairs)", "3-5", "6-10", "11-100", "101-1000", "1001-10000", ">10000"]
		bracket_order = [b for b in bracket_order if b in brackets]
		
		debug_print("Component count by size bracket:", debug)
		for bracket in bracket_order:
			if bracket in brackets:
				debug_print(f"  {bracket}: {brackets[bracket]:,} components", debug)
		
		isolated_nodes = size_distribution.get(1, 0)
		isolated_percent = (isolated_nodes / num_components) * 100 if num_components > 0 else 0
		
		small_components = sum(count for size, count in size_distribution.items() if size <= 5)
		small_percent = (small_components / num_components) * 100 if num_components > 0 else 0
		
		debug_print(f"\nIsolated nodes: {isolated_nodes:,} ({isolated_percent:.2f}% of components)", debug)
		debug_print(f"Small components (≤5 nodes): {small_components:,} ({small_percent:.2f}% of components)", debug)
		
		max_possible_edges = total_nodes * (total_nodes - 1) / 2
		density = edges_count / max_possible_edges if max_possible_edges > 0 else 0
		debug_print(f"\nGraph density: {density:.8f}", debug)
		
		lcc_ratio = (largest_component_size / total_nodes) * 100 if total_nodes > 0 else 0
		debug_print(f"\nLCC contains {largest_component_size:,} nodes ({lcc_ratio:.2f}% of the graph)", debug)
		
		lcc_density = lcc_edges_count / (len(lcc_nodes) * (len(lcc_nodes) - 1) / 2) if len(lcc_nodes) > 1 else 0
		debug_print(f"LCC density: {lcc_density:.8f}", debug)
		
		execution_time = time.time() - start_time
		debug_print(f"\nAnalysis completed in {execution_time:.2f} seconds", debug)
	
	return edges_count-lcc_edges_count

def extract_largest_connected_component(tsv_file, output_file, debug=True):
	"""
	Main function to analyze a graph from a TSV file.
	
	Args:
		tsv_file: Path to the TSV file
		output_file: output file
		debug: If True, show debug messages
	"""
	if not os.path.isfile(tsv_file):
		print(f"Error: file '{tsv_file}' not found.")
		return None
	
	res = count_connected_components(
		tsv_file=tsv_file,
		output_file=output_file,
		debug=debug,
	)

	return res

# Utils for model training 

def get_entity_type(entity):
	"""Extract entity type from entity ID (e.g., 'Compound::DB00001' -> 'Compound').
	If no '::' separator is found, returns 'Entity' as default type.
	"""
	if "::" in str(entity):
		return str(entity).split("::")[0]
	return "Entity"

def get_edge_type(head, tail):
	"""Derive edge type from head and tail entity types.
	E.g., 'Compound::DB00001' + 'ExtGene::Rv0001' -> 'Compound-ExtGene'
	"""
	return f"{get_entity_type(head)}-{get_entity_type(tail)}"

def load_data(edge_index_path, features_paths_per_type, quiet=True, debug=False, 
			  head_col=None, interaction_col=None, tail_col=None):
	"""
	Load edge index and node features from a TSV file.
	
	Args:
		edge_index_path: Path to the TSV file containing the edge index.
		features_paths_per_type: Dictionary mapping node types to feature file paths.
		quiet: If True, suppress output messages.
		debug: If True, save debug files.
		head_col: Column name for head entities (default: auto-detect from first 3 columns).
		interaction_col: Column name for interaction/relation (default: auto-detect).
		tail_col: Column name for tail entities (default: auto-detect).
	
	Returns:
		Tuple of (edge_index DataFrame, node_features dictionary).
		
	Notes:
		- The 'type' column is NOT used by networks. Edge types are derived on-the-fly
		  from entity prefixes (e.g., "Compound::xyz" -> "Compound").
		- Column names can be specified explicitly or auto-detected from first 3 columns.
	"""
	if edge_index_path.endswith(".zip"):
		edge_ind = pd.read_csv(edge_index_path, sep='\t', dtype=str, compression='zip')
	else:
		edge_ind = pd.read_csv(edge_index_path, sep='\t', dtype=str)	
	
	# Auto-detect or use specified column names for head, interaction, tail
	# Assume the first 3 columns are head, interaction, tail if not specified
	columns = edge_ind.columns.tolist()
	# dedux the datsaframe at the first three columns
	edge_ind = edge_ind.iloc[:, :3]
	
	if head_col is None:
		head_col = columns[0] if len(columns) >= 1 else 'head'
	if interaction_col is None:
		interaction_col = columns[1] if len(columns) >= 2 else 'interaction'
	if tail_col is None:
		tail_col = columns[2] if len(columns) >= 3 else 'tail'
	
	# Rename columns to standard names if different
	rename_map = {}
	if head_col != 'head':
		rename_map[head_col] = 'head'
	if interaction_col != 'interaction':
		rename_map[interaction_col] = 'interaction'
	if tail_col != 'tail':
		rename_map[tail_col] = 'tail'
	
	if rename_map:
		edge_ind = edge_ind.rename(columns=rename_map)
		if not quiet:
			print(f"[load_data] Renamed columns: {rename_map}")
	
	# Keep only the triple columns (head, interaction, tail) - ignore any 'type', 'source', etc.
	edge_ind = edge_ind[['head', 'interaction', 'tail']].copy()
	
	node_features = {}
	all_edge_ind_entities = set(edge_ind["head"]).union(set(edge_ind["tail"]))

	if features_paths_per_type != None: 
		node_features = {node_type: pd.read_csv(feature_path).drop_duplicates() for node_type, feature_path in features_paths_per_type.items()}
		
	entities_types = set(edge_ind["head"].apply(lambda x: x.split("::")[0])).union(set(edge_ind["tail"].apply(lambda x: x.split("::")[0]))) 
	for node_type in entities_types:
		if node_type not in node_features:
			node_features[node_type] = None
		else:
			node_features[node_type] = node_features[node_type][node_features[node_type]["id"].isin(all_edge_ind_entities)]  ## filter the entities not present in the edge index
			
	triplets_count = len(edge_ind)
	# print(f"Tripelt Count: {triplets_count}")
	interaction_types_count = edge_ind["interaction"].nunique()

	if not quiet:
		print(colored(f'[loaded edge index] triplets count: {triplets_count} interaction types count: {interaction_types_count}', 'green'))
	if debug:
		os.makedirs("debug", exist_ok=True)
		edge_ind.to_csv("debug/edge_index.csv", index=False)
		# save node type in json
		with open("debug/node_features.json", "w") as f:
			json.dump({node_type: features_path for node_type, features_path in features_paths_per_type.items()}, f, indent=4)

	return edge_ind, node_features

def entities2id(edge_index, node_features_per_type):
	# create a dictionary that maps the entities to an integer id
	entities = set(edge_index["head"]).union(set(edge_index["tail"]))
	entities2id = {}
	all_nodes_per_type = {}
	for x in entities:
		if x.split("::")[0] not in all_nodes_per_type:
			all_nodes_per_type[x.split("::")[0]] = [x]
		else:
			all_nodes_per_type[x.split("::")[0]].append(x)
	for node_type, features in node_features_per_type.items():
		if features is None:
			for idx, node in enumerate(all_nodes_per_type[node_type]):
				entities2id[node] = idx #+ offset
			continue
		for idx, node in enumerate(features.id):
			entities2id[node] = idx #+ offset
	return entities2id, all_nodes_per_type

def entities2id_offset(edge_index, node_features_per_type, quiet=False):
	# create a dictionary that maps the entities to an integer id
	entities = sorted(set(edge_index["head"]).union(set(edge_index["tail"])))
	entities2id = {}
	all_nodes_per_type = {}
	
	for x in entities:
		if x.split("::")[0] not in all_nodes_per_type:
			all_nodes_per_type[x.split("::")[0]] = [x]
		else:
			all_nodes_per_type[x.split("::")[0]].append(x)

	# Ensure deterministic node ordering per type.
	for node_type in all_nodes_per_type:
		all_nodes_per_type[node_type] = sorted(all_nodes_per_type[node_type])
	
	if not quiet:
		for node_type in sorted(all_nodes_per_type):
			nodes = all_nodes_per_type[node_type]
			print(colored(f'	[{node_type}] count: {len(nodes)}', 'green'))

	offset = 0
	for node_type in sorted(node_features_per_type):
		features = node_features_per_type[node_type]
		if features is None:
			for idx, node in enumerate(all_nodes_per_type[node_type]):
				entities2id[node] = idx + offset
			offset += len(all_nodes_per_type[node_type])
			continue
		
		all_edge_index_nodes = sorted([x for x in features.id.values if x in all_nodes_per_type[node_type]])

		for idx, node in enumerate(all_edge_index_nodes):
			entities2id[node] = idx + offset
		offset += len(all_edge_index_nodes)

	return entities2id, all_nodes_per_type

def rel2id(edge_index):
	# create a dictionary that maps the relations to an integer id
	rel2id = {rel.replace(" ","_"): idx for idx, rel in enumerate(edge_index.interaction.unique())}
	relations = list(rel2id.keys())
	for rel in relations:
		rel2id[f"rev_{rel}"] = rel2id[rel] 
	return rel2id

def rel2id_offset(edge_index):
	# create a dictionary that maps the relations to an integer id
	relation2id = {rel.replace(" ","_"): idx for idx, rel in enumerate(edge_index.interaction.unique())}
	rel_number = len(relation2id)
	relations = list(relation2id.keys())

	for rel in relations:
		relation2id[f"rev_{rel}"] = relation2id[rel] + rel_number

	relation2id["self"] = rel_number*2 
	# print(relation2id)
	return relation2id

def index_entities_edge_ind(edge_ind, entities2id):
	# create a new edge index where the entities are replaced by their integer id
	indexed_edge_ind = edge_ind.copy()
	indexed_edge_ind["head"] = indexed_edge_ind["head"].apply(lambda x: entities2id[x])
	indexed_edge_ind["tail"] = indexed_edge_ind["tail"].apply(lambda x: entities2id[x])
	return indexed_edge_ind

def edge_ind_to_id(edge_ind, entities2id, relation2id):
	# create a new edge index where the entities are replaced by their integer id
	indexed_edge_ind = edge_ind.copy()
	indexed_edge_ind["head"] = indexed_edge_ind["head"].apply(lambda x: entities2id[x])
	indexed_edge_ind["interaction"] = indexed_edge_ind["interaction"].apply(lambda x: relation2id[x.replace(" ","_")])
	indexed_edge_ind["tail"] = indexed_edge_ind["tail"].apply(lambda x: entities2id[x])
	return indexed_edge_ind

def graph_to_undirect(edge_index, rel_num):
	reverse_triplets = edge_index.copy()
	reverse_triplets[:,[0,2]] = reverse_triplets[:,[2,0]]
	reverse_triplets[:,1] += rel_num//2
	undirected_edge_index = np.concatenate([edge_index, reverse_triplets], axis=0)
	return torch.tensor(undirected_edge_index)

def add_self_loops(train_index, num_entities, num_relations):
	# In `rel2id_offset`, the self-loop relation id is the last one: `num_relations - 1`.
	# Using `num_relations` would create an out-of-range relation id.
	head = torch.tensor([x for x in range(num_entities)])
	interaction = torch.tensor([num_relations - 1 for _ in range(num_entities)])
	tail = torch.tensor([x for x in range(num_entities)])
	self_loops = torch.cat([head.view(1,-1), interaction.view(1,-1), tail.view(1,-1)], dim=0).T
	train_index_self_loops = torch.cat([train_index, self_loops], dim=0)
	return train_index_self_loops

def set_target_label(edge_ind, target_edges, debug=False):
	"""
	Label edges as target (1) or non-target (0) based on task specification.
	
	The task can be specified as:
	  1. An edge type derived from entity prefixes (e.g., 'Compound-ExtGene')
	  2. An interaction name from the dataset (e.g., 'TARGET', 'GENE_BIND', 'CMP_BIND')
	
	Both are matched case-insensitively and with underscore/space normalization.
	
	Args:
		edge_ind: DataFrame with 'head', 'interaction', 'tail' columns.
		target_edges: List of tasks (edge types or interaction names).
		debug: If True, save debug files.
	
	Returns:
		DataFrame with added 'label' column (1 for target, 0 otherwise).
	"""
	def normalize(s):
		"""Normalize string for matching: lowercase, replace spaces with underscores"""
		return str(s).lower().replace(" ", "_")
	
	# Build normalized set of targets (including reversed edge types)
	target_set_normalized = set()
	for t in target_edges:
		target_set_normalized.add(normalize(t))
		# Add reversed version for edge types: 'A-B' -> 'B-A'
		if '-' in t:
			parts = t.split('-')
			if len(parts) == 2:
				target_set_normalized.add(normalize(f"{parts[1]}-{parts[0]}"))
	
	# Derive edge type from entity prefixes
	edge_ind["_edge_type"] = edge_ind.apply(
		lambda row: get_edge_type(row['head'], row['tail']),
		axis=1
	)
	
	# Normalize interaction column for matching
	edge_ind["_interaction_norm"] = edge_ind["interaction"].apply(normalize)
	edge_ind["_edge_type_norm"] = edge_ind["_edge_type"].apply(normalize)
	
	# Match against either edge type OR interaction name
	edge_ind["label"] = (
		edge_ind["_edge_type_norm"].isin(target_set_normalized) |
		edge_ind["_interaction_norm"].isin(target_set_normalized)
	).astype(int)
	
	# Show available options for debugging
	available_edge_types = edge_ind["_edge_type"].unique().tolist()
	available_interactions = edge_ind["interaction"].unique().tolist()
	num_target = edge_ind["label"].sum()
	
	print(f"[set_target_label] Looking for: {target_edges}")
	print(f"[set_target_label] Available edge types: {available_edge_types}")
	print(f"[set_target_label] Available interactions: {available_interactions}")
	print(f"[set_target_label] Found {num_target} target edges out of {len(edge_ind)} total")
	
	# Remove temporary columns
	edge_ind = edge_ind.drop(columns=["_edge_type", "_interaction_norm", "_edge_type_norm"])
	
	if debug:
		os.makedirs("debug", exist_ok=True)
		edge_ind.to_csv("debug/edge_index_with_labels.csv", index=False)

	return edge_ind

def select_target_triplets(edge_index):
	target_triplets = edge_index.loc[edge_index["label"]==1,:].copy()
	non_target_triplets = edge_index.loc[edge_index["label"]==0,:].copy()
	return non_target_triplets, target_triplets

def negative_sampling(target_triplets, negative_rate=1):
	target_triplets = np.array(target_triplets)
	negative_rate = int(negative_rate)
	src, _, dst = target_triplets.T
	uniq_entity = np.unique((src, dst))
	pos_num = target_triplets.shape[0]
	neg_num = pos_num * negative_rate
	neg_samples = np.tile(target_triplets, (negative_rate, 1))
	values = np.random.choice(uniq_entity, size=neg_num)
	choices = np.random.uniform(size=neg_num)
	# choice on who to perturb
	subj = choices > 0.5
	obj = choices <= 0.5
	# actual perturbation
	neg_samples[subj, 0] = values[subj]
	neg_samples[obj, 2] = values[obj]
	labels = torch.zeros(target_triplets.shape[0]+neg_samples.shape[0])
	labels[:target_triplets.shape[0]] = 1
	neg_samples = torch.tensor(neg_samples)
	samples = torch.cat([torch.tensor(target_triplets), neg_samples], dim=0)
	return samples, labels

def negative_sampling_filtered_orignal(
	target_triplets,
	negative_rate=1,
	all_true_triplets=None,
	num_entities=None,
	seed=42,
	max_attempts_per_negative=50,
	debug=False
):
	"""
	Alternative negative sampling for KG link prediction.

	Compared to `negative_sampling`, this version:
	- filters false negatives using `all_true_triplets` (if provided),
	- supports deterministic sampling through `seed`,
	- accepts float/int `negative_rate` safely.

	Args:
		target_triplets: Positive triplets (N, 3) [h, r, t].
		negative_rate: Number of negatives per positive.
		all_true_triplets: Optional iterable with all true KG triplets to filter against.
		num_entities: Optional total number of entities; if None, infer from data.
		seed: RNG seed for reproducibility.
		max_attempts_per_negative: Max retries to avoid sampling a true triplet.

	Returns:
		samples: Tensor of shape (N + N*negative_rate, 3).
		labels: Tensor of shape (N + N*negative_rate,), positives first.
	"""
	target_triplets = np.asarray(target_triplets, dtype=np.int64)
	if target_triplets.ndim != 2 or target_triplets.shape[1] != 3:
		raise ValueError("target_triplets must have shape (N, 3)")
	if target_triplets.shape[0] == 0:
		return torch.empty((0, 3), dtype=torch.long), torch.empty((0,), dtype=torch.float)

	neg_rate = int(negative_rate)
	if neg_rate <= 0:
		pos_tensor = torch.tensor(target_triplets, dtype=torch.long)
		labels = torch.ones(pos_tensor.shape[0], dtype=torch.float)
		return pos_tensor, labels

	rng = np.random.default_rng(seed)
	pos_num = target_triplets.shape[0]
	neg_num = pos_num * neg_rate

	if num_entities is None:
		src, _, dst = target_triplets.T
		unique_entities = np.unique(np.concatenate([src, dst]))
	else:
		unique_entities = np.arange(int(num_entities), dtype=np.int64)

	true_set = set(map(tuple, target_triplets.tolist()))
	if all_true_triplets is not None:
		all_true_arr = np.asarray(all_true_triplets, dtype=np.int64)
		if all_true_arr.ndim == 2 and all_true_arr.shape[1] == 3:
			true_set.update(map(tuple, all_true_arr.tolist()))

	neg_samples = np.empty((neg_num, 3), dtype=np.int64)
	filled = 0
	for i in range(pos_num):
		h, r, t = target_triplets[i]
		for _ in range(neg_rate):
			candidate = None
			for _attempt in range(max_attempts_per_negative):
				if rng.random() > 0.5:
					candidate = (int(rng.choice(unique_entities)), int(r), int(t))
				else:
					candidate = (int(h), int(r), int(rng.choice(unique_entities)))
				if candidate not in true_set:
					break
			if candidate is None:
				candidate = (int(h), int(r), int(t))
			neg_samples[filled] = candidate
			filled += 1

	samples = torch.cat(
		[
			torch.tensor(target_triplets, dtype=torch.long),
			torch.tensor(neg_samples, dtype=torch.long),
		],
		dim=0
	)
	labels = torch.zeros(samples.shape[0], dtype=torch.float)
	labels[:pos_num] = 1.0
	if debug:
		print(f"Generated {neg_num} negative samples for {pos_num} positives.")
	return samples, labels


def negative_sampling_filtered(
    target_triplets,
    all_entities,           # tutti i nodi del grafo (array di id)
    negative_rate=1,
    all_true_triplets=None, # tutti i 4.1M edge per filtrare falsi negativi
    seed=42,
    max_attempts_per_negative=50,
    debug=False
):
    """
    Correct negative sampling for KG link prediction.

    Key fixes vs previous versions:
    - uses all_entities (full graph node set) instead of inferring from target_triplets
    - filters false negatives using all_true_triplets
    - type-constrained: only perturbs head with valid heads, tail with valid tails
    """
	

    target_triplets = np.asarray(target_triplets, dtype=np.int64)
    if target_triplets.ndim != 2 or target_triplets.shape[1] != 3:
        raise ValueError("target_triplets must have shape (N, 3)")
    if target_triplets.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.long), torch.empty((0,), dtype=torch.float)

    neg_rate = int(negative_rate)
    rng = np.random.default_rng(seed)
    pos_num = target_triplets.shape[0]
    neg_num = pos_num * neg_rate

    # --- FIX 1: usa tutte le entità del grafo ---
    all_entities = np.asarray(all_entities, dtype=np.int64)

    # --- FIX 2: costruisci il set di tutti i triplet veri per filtrare falsi negativi ---
    true_set = set(map(tuple, target_triplets.tolist()))
    if all_true_triplets is not None:
        all_true_arr = np.asarray(all_true_triplets, dtype=np.int64)
        if all_true_arr.ndim == 2 and all_true_arr.shape[1] == 3:
            true_set.update(map(tuple, all_true_arr.tolist()))

    # --- FIX 3: type-constrained sampling ---
    # Per relazioni Compound-TARGET-ExtGene ha senso perturbare:
    # - la testa solo con altri Compound
    # - la coda solo con altri ExtGene
    # Inferisce i candidati validi per testa e coda dalla relazione
    heads_by_rel = {}
    tails_by_rel = {}
    if all_true_triplets is not None:
        all_true_arr = np.asarray(all_true_triplets, dtype=np.int64)
        for h, r, t in all_true_arr:
            heads_by_rel.setdefault(int(r), set()).add(int(h))
            tails_by_rel.setdefault(int(r), set()).add(int(t))
    for h, r, t in target_triplets:
        heads_by_rel.setdefault(int(r), set()).add(int(h))
        tails_by_rel.setdefault(int(r), set()).add(int(t))
    # converti in array per np.choice
    heads_by_rel = {r: np.array(list(v), dtype=np.int64) for r, v in heads_by_rel.items()}
    tails_by_rel = {r: np.array(list(v), dtype=np.int64) for r, v in tails_by_rel.items()}

    neg_samples = np.empty((neg_num, 3), dtype=np.int64)
    filled = 0

    for i in range(pos_num):
        h, r, t = target_triplets[i]
        r_int = int(r)

        # candidati type-constrained, fallback a all_entities
        head_candidates = heads_by_rel.get(r_int, all_entities)
        tail_candidates = tails_by_rel.get(r_int, all_entities)

        for _ in range(neg_rate):
            candidate = None
            for _attempt in range(max_attempts_per_negative):
                if rng.random() > 0.5:
                    new_h = int(rng.choice(head_candidates))
                    candidate = (new_h, r_int, int(t))
                else:
                    new_t = int(rng.choice(tail_candidates))
                    candidate = (int(h), r_int, new_t)

                if candidate not in true_set:
                    break
                candidate = None  # era un falso negativo, riprova

            if candidate is None:
                # fallback: usa il positivo stesso (non ideale ma evita crash)
                # in pratica non dovrebbe mai accadere con pool grandi
                candidate = (int(h), r_int, int(t))
                if debug:
                    print(f"[WARN] Could not find valid negative for triplet {i} after {max_attempts_per_negative} attempts")

            neg_samples[filled] = candidate
            filled += 1

    samples = torch.cat([
        torch.tensor(target_triplets, dtype=torch.long),
        torch.tensor(neg_samples, dtype=torch.long),
    ], dim=0)

    labels = torch.zeros(samples.shape[0], dtype=torch.float)
    labels[:pos_num] = 1.0


    if debug:
        print(f"[negative_sampling] {pos_num} positives, {neg_num} negatives generated")
        print(f"  Total unique entities: {len(all_entities)}")
        print(f"  Head candidates for rel {r_int}: {len(head_candidates)}")
        print(f"  Tail candidates for rel {r_int}: {len(tail_candidates)}")

    return samples, labels



def triple_sampling_basic(target_triplet, val_size, test_size, quiet=True, seed=42):
	val_len = len(target_triplet) * val_size
	# split the data into training, testing, and validation 
	temp_data, test_data = train_test_split(target_triplet, test_size=test_size, random_state=seed, shuffle=True)
	train_data, val_data = train_test_split(temp_data, test_size=(val_len / len(temp_data)), random_state=seed, shuffle=True)
	# print the shapes of the resulting sets
	if not quiet:
		print(f"Total number of target edges: {len(target_triplet)}")
		print(f"\tTraining set shape: {len(train_data)}")
		print(f"\tValidation set shape: {len(val_data)}" )
		print(f"\tTesting set shape: {len(test_data)}\n")
	return train_data, val_data, test_data


def triple_sampling(target_triplet, val_size, test_size, quiet=True, seed=42):
    """
    Split stratificato per gene (coda della relazione TARGET).
    Garantisce che ogni gene con >= 2 edge abbia almeno un edge nel train.
    Geni con un solo edge vanno sempre nel train.

	La prima è attesa e corretta: lo split stratificato ha messo nel test set 
	solo geni con ≥2 TARGET edges, eliminando i casi "facili" dove il modello 
	poteva sfruttare geni molto connessi. Il task è genuinamente più difficile.
    """
    target_triplet = list(target_triplet)
    
    # raggruppa per gene (tail = colonna 2)
    from collections import defaultdict
    tail_to_triplets = defaultdict(list)
    for triplet in target_triplet:
        tail = triplet[2]  # ExtGene id
        tail_to_triplets[tail].append(triplet)
    
    train_data, val_data, test_data = [], [], []

    for tail, triplets in tail_to_triplets.items():
        if len(triplets) == 1:
            # geni con un solo edge: sempre in train, non valutabili
            train_data.extend(triplets)
        elif len(triplets) == 2:
            # uno in train, uno in test
            train_data.append(triplets[0])
            test_data.append(triplets[1])
        else:
            # split normale ma garantendo almeno 1 in train
            temp, test = train_test_split(triplets, test_size=test_size, random_state=seed)
            if len(temp) == 1:
                train_data.extend(temp)
            else:
                val_len = max(1, round(len(temp) * val_size))
                train, val = train_test_split(temp, test_size=val_len/len(temp), random_state=seed)
                train_data.extend(train)
                val_data.extend(val)
            test_data.extend(test)

    if not quiet:
        print(f"Total number of target edges: {len(target_triplet)}")
        print(f"\tTraining set shape: {len(train_data)}")
        print(f"\tValidation set shape: {len(val_data)}")
        print(f"\tTesting set shape: {len(test_data)}\n")
        
        # mostra quanti geni sono solo in train
        train_tails = set(t[2] for t in train_data)
        test_tails = set(t[2] for t in test_data)
        only_train = train_tails - test_tails
        print(f"\tGeni solo in train (non valutabili): {len(only_train)}")

    return train_data, val_data, test_data




def flat_index(triplets, num_nodes):
	fr, to = triplets[:, 0]*num_nodes, triplets[:, 2]
	offset = triplets[:, 1] * num_nodes*num_nodes 
	flat_indices = fr + to + offset
	return flat_indices

def entities_features_flattening(node_features_per_type, all_nodes_per_type):
	# flatten the features of the entities
	flattened_features_per_type = {}

	for node_type, features in node_features_per_type.items():
		if features is None:
			flattened_features_per_type[node_type] = None # torch.ones((len(all_nodes_per_type[node_type]), 1),dtype=torch.float)
			continue
		features = features.drop(columns=["id"])
		features = features.map(lambda x: np.array([int(v) for v in x ]))
		features_matrix = []
		for x in features.values:
			features_matrix.append(np.concatenate(x))
		flattened_features_per_type[node_type] = torch.tensor(np.array(features_matrix), dtype=torch.float)

	return flattened_features_per_type

def create_hetero_data(indexed_edge_ind, node_features_per_type, rel2id, verbose=True):
	data = HeteroData()
	total_nodes = 0
	for node_type, features in node_features_per_type.items():
		data[f"{node_type}"].x = torch.tensor(features, dtype=torch.float).contiguous()
		total_nodes += len(features)
	all_interaction_per_type = indexed_edge_ind[["interaction","type"]].drop_duplicates().values
	for interaction, entities in all_interaction_per_type:
		edge_interaction = indexed_edge_ind.loc[(indexed_edge_ind["interaction"] == interaction) & (indexed_edge_ind["type"]==entities)]
		entity_types = entities.split(" - ")  ######## "-" or " - " depending on the dataset
		edges = edge_interaction.loc[:,["head","tail"]].values
		data[entity_types[0].replace(" ",""),interaction.replace(" ","_"),entity_types[1].replace(" ","")].edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

	return data

def data_split_and_negative_sampling( data, target_edges, rev_target, val_ratio=0.2, test_ratio=0.3 ,neg_sampling_ratio=1.0):
	transform_split = T.RandomLinkSplit(
		num_val=val_ratio,
		num_test=test_ratio,
		neg_sampling_ratio=neg_sampling_ratio,
		add_negative_train_samples=True,
		is_undirected=True,
		edge_types=target_edges,
		rev_edge_types=rev_target
	)
	return transform_split(data)

def get_all_triplets(data, rel2id):	
	head_tail = torch.cat(list(data.edge_index_dict.values()), dim=1)
	# add the relation id to the triplets TO the last dimension
	rel_ids = []
	for edge_type in data.edge_types:
		rel_ids.append(torch.tensor([rel2id[edge_type[1]] for _ in range(data[edge_type].edge_index.shape[1])]))
	rel_ids = torch.cat(rel_ids, dim=0)
	triplets = torch.cat([head_tail, rel_ids.view(1,-1)], dim=0)
	triplets = triplets.T
	# Swap the order of tail and interaction to get the triplets in the form (head, interaction, tail)
	triplets[:,[1,2]] = triplets[:,[2,1]]
	return triplets

def get_target_triplets_and_labels(data, target_edges, relation2id):
	all_target_triplets = []
	all_labels = []

	for target_edge in target_edges:

		head_tail = data[target_edge].edge_label_index
		rel_ids = torch.tensor([relation2id[target_edge[1]] for _ in range(head_tail.shape[1])])
		# add the relation id to the triplets TO the last dimension

		triplets = torch.cat([head_tail, rel_ids.view(1,-1)], dim=0)
		triplets = triplets.T
		# Swap the order of tail and interaction to get the triplets in the form (head, interaction, tail)
		triplets[:,[1,2]] = triplets[:,[2,1]]
		all_target_triplets.append(triplets)
		all_labels.append(data[target_edge].edge_label)

	all_target_triplets = torch.cat(all_target_triplets, dim=0)
	all_labels = torch.cat(all_labels, dim=0)
	
	return all_target_triplets, all_labels

def graph_transform(data):
	transformation = []
	transformation.append(T.ToUndirected())
	transformation.append(T.AddSelfLoops())
	transformation.append(T.RemoveDuplicatedEdges()) # Always remove duplicated edges
	transform = T.Compose(transformation)
	data = transform(data)
	return data

# ======== EVALUATION METRICS ========
"""
Evaluation modes:
- Sampled: ranks the positive against a random pool of `num_generate` candidates.
  Fast but optimistic; not comparable with published benchmarks.
- Filtered (standard in KGE literature): ranks against ALL entities in the graph,
  masking known true triples. This is the evaluation used in the paper.
"""
### Sampled evaluation (legacy, not used in the paper)

def evaluation_metrics_legacy(model, embeddings, all_target_triplets, test_triplet, num_generate, device, hits=[1,3,10]):
	"""
    Approximate (sampled) MRR and Hits@k evaluation for KG link prediction.

    For each test triplet (h,r,t), ranks the true entity against a randomly
    sampled subset of candidate entities (size = num_generate) instead of the
    full entity set. Both head and tail prediction are evaluated by replacing
    h or t with sampled nodes and computing DistMult scores.

    This produces a bidirectional *sampled ranking* estimate of MRR/Hits,
    which is faster and suitable for training monitoring but optimistic and
    dependent on the sampling size. Results are not directly comparable to
    standard full-ranking KG benchmarks.
    """
	src, _, dst = all_target_triplets.T
	
	unique_nodes = torch.unique(torch.cat((src,dst), dim = 0))
	if num_generate > unique_nodes.size(0):
		print(f"[ERROR] requested more triplets than available nodes")
	with torch.no_grad():
		for head in [True, False]:
			generator = torch.Generator().manual_seed(42)
			random_indices = torch.randperm(unique_nodes.size(0), generator=generator)[:num_generate]
			selected_nodes = unique_nodes[random_indices]
			if head:
				head_rel = test_triplet[:, :2] #(test_triplet)  (all_target_triplets)
				head_rel = torch.repeat_interleave(head_rel, num_generate, dim=0) # shape (test_triplet.size(0)*100, 3)
				target_tails = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1,1) #shape (test_triplet.size(0)*100, 1)
				mrr_triplets = torch.cat((head_rel, target_tails), dim=-1) #shape (test_triplet.size(0)*100, 3)
			else:
				rel_tail = test_triplet[:, 1:]
				rel_tail = torch.repeat_interleave(rel_tail, num_generate, dim=0) # shape (test_triplet.size(0)*100, 3)
				target_heads = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1,1) #shape (test_triplet.size(0)*100, 1)
				mrr_triplets = torch.cat((target_heads, rel_tail), dim=-1) #shape (test_triplet.size(0)*100, 3)
			mrr_triplets = mrr_triplets.view(test_triplet.size(0), num_generate, 3)# shape(test triplets, mrr_triplets, 3)
			mrr_triplets = torch.cat((mrr_triplets, test_triplet.view(-1,1,3)), dim=1)# shape(test triplets, mrr_triplets+1, 3)
			scores = model.distmult(embeddings, mrr_triplets.view(-1,3)).view(test_triplet.size(0), num_generate+1)
			_, ranks = torch.sort(scores, descending=True)
			if head:
				ranks_s =  ranks[:, -1]
			else:
				ranks_o =  ranks[:, -1]
		# rank can't be zero since we then take the reciprocal of 0, so everyone is shifted by one position
		ranks = torch.cat([ranks_s, ranks_o]) + 1 # change to 1-indexed
		mrr = torch.mean(1.0 / ranks)
		hits = {at:0 for at in hits}
		for hit in hits:
			avg_count = torch.sum((ranks <= hit))/ranks.size(0)
			hits[hit] = avg_count.item()
	return mrr.item(), hits #, auroc, auprc


def evaluation_metrics_sampled(model, embeddings, all_target_triplets, test_triplet, num_generate, device, hits=[1,3,10], standardized_negatives=False):
    """
    Approximate (sampled) MRR and Hits@k evaluation for KG link prediction.

    For each test triplet (h,r,t), ranks the true entity against a randomly
    sampled subset of candidate entities (size = num_generate) instead of the
    full entity set. Both head and tail prediction are evaluated by replacing
    h or t with sampled nodes and computing DistMult scores.

    This produces a bidirectional *sampled ranking* estimate of MRR/Hits,
    which is faster and suitable for training monitoring but optimistic and
    dependent on the sampling size. Results are not directly comparable to
    standard full-ranking KG benchmarks.

    FIXES rispetto alla versione originale:
      1. Traccia la posizione del positivo dopo il sort (prima assumeva ranks[:,-1])
      2. Clamp num_generate per evitare errore se > nodi disponibili
      3. Gestisce il caso hits passato come dict (re-init sicuro)

    Args:
        standardized_negatives: se True, campiona i negativi da tutti gli entity ID
            (torch.arange su embeddings) invece che dai soli nodi presenti in
            all_target_triplets. Garantisce lo stesso set di negativi tra varianti
            di ablation study (a parità di num_generate e entity set), rendendo
            le metriche direttamente confrontabili.
    """
    if standardized_negatives:
        unique_nodes = torch.arange(embeddings.size(0), device=device)
    else:
        src, _, dst = all_target_triplets.T
        unique_nodes = torch.unique(torch.cat((src, dst), dim=0))

    # Clamp: non possiamo generare più candidati di quanti nodi abbiamo
    if num_generate > unique_nodes.size(0):
        print(f"[WARN] num_generate ({num_generate}) > unique nodes ({unique_nodes.size(0)}), clamping.")
        num_generate = unique_nodes.size(0)

    # Re-init hits come dict pulito (evita side-effect se passato come default mutable)
    hits_k = list(hits.keys()) if isinstance(hits, dict) else list(hits)
    
    # Indice del positivo: è sempre l'ultimo concatenato -> posizione num_generate
    positive_idx = num_generate

    with torch.no_grad():
        for head in [True, False]:
            generator = torch.Generator().manual_seed(42)
            random_indices = torch.randperm(unique_nodes.size(0), generator=generator)[:num_generate]
            selected_nodes = unique_nodes[random_indices]

            if head:
                # Tail prediction: fisso (h, r), vario t
                head_rel = test_triplet[:, :2]
                head_rel = torch.repeat_interleave(head_rel, num_generate, dim=0)
                target_tails = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1, 1)
                mrr_triplets = torch.cat((head_rel, target_tails), dim=-1)
            else:
                # Head prediction: fisso (r, t), vario h
                rel_tail = test_triplet[:, 1:]
                rel_tail = torch.repeat_interleave(rel_tail, num_generate, dim=0)
                target_heads = torch.tile(selected_nodes, (1, test_triplet.size(0))).view(-1, 1)
                mrr_triplets = torch.cat((target_heads, rel_tail), dim=-1)

            # Shape: (num_test, num_generate, 3)
            mrr_triplets = mrr_triplets.view(test_triplet.size(0), num_generate, 3)
            # Concatena il positivo come ultimo candidato -> indice num_generate
            mrr_triplets = torch.cat((mrr_triplets, test_triplet.view(-1, 1, 3)), dim=1)

			# Score: (num_test, num_generate+1)
            # scores = model.distmult(embeddings, mrr_triplets.view(-1, 3)).view(test_triplet.size(0), num_generate + 1)

            # Score: (num_test, num_generate+1)  — chunked to avoid OOM on large val sets
            flat_triplets = mrr_triplets.view(-1, 3)
            SCORE_CHUNK = 100_000
            if flat_triplets.size(0) <= SCORE_CHUNK:
                scores = model.distmult(embeddings, flat_triplets)
            else:
                scores = torch.cat([
                    model.distmult(embeddings, flat_triplets[i:i + SCORE_CHUNK])
                    for i in range(0, flat_triplets.size(0), SCORE_CHUNK)
                ])
            scores = scores.view(test_triplet.size(0), num_generate + 1)

            # Sort decrescente: ranks[i,j] = indice originale del candidato in posizione j
            _, sorted_indices = torch.sort(scores, descending=True)

            # FIX: trova la posizione del positivo (indice positive_idx) nel ranking ordinato
            # Per ogni riga, cerca dove sorted_indices == positive_idx
            # (sorted_indices == positive_idx) è un bool tensor (num_test, num_generate+1)
            # .nonzero() restituisce le coordinate [riga, colonna] dei True
            # La colonna è la posizione nel ranking (0-indexed)
            positive_positions = (sorted_indices == positive_idx).nonzero(as_tuple=False)[:, 1]

            if head:
                ranks_s = positive_positions
            else:
                ranks_o = positive_positions

        # +1 per passare a 1-indexed (rank 1 = migliore)
        ranks = torch.cat([ranks_s, ranks_o]).float() + 1

        mrr = torch.mean(1.0 / ranks)

        hits_result = {}
        for k in hits_k:
            avg_count = torch.sum((ranks <= k)).float() / ranks.size(0)
            hits_result[k] = avg_count.item()

    return mrr.item(), hits_result

# ======== FULL RANKING EVALUATION: MRR e Hits@k su TUTTI i nodi del grafo (filtered, confrontabile con la letteratura) ========

def evaluation_metrics_full(model, embeddings, all_graph_nodes, test_triplet, device, hits=[1,3,10]):
    """
    MRR e Hits calcolati su tutti i nodi del grafo.
    Da chiamare SEPARATAMENTE dopo evaluation_metrics originale.
    Non sostituisce nulla — è additive.
    """"""
    Full-entity MRR and Hits@k evaluation for KG link prediction.

    For each test triplet (h,r,t), ranks the true tail entity against all
    entities in the graph by scoring (h,r,e) for every e ∈ E. This corresponds
    to the standard *full ranking* KG evaluation protocol (deterministic and
    sampling-free), producing the true global rank of the positive triple.

    This implementation evaluates tail prediction only (unidirectional) and
    therefore yields lower but benchmark-comparable metrics compared to
    sampled evaluation.
    """
    model.eval()
    unique_nodes = all_graph_nodes.to(device)  # torch.arange(num_entities).to(device)
    num_generate = unique_nodes.size(0)
    
    ranks_list = []
    
    with torch.no_grad():
        for i in range(test_triplet.size(0)):
            triplet = test_triplet[i]  # (h, r, t)
            h, r, t = triplet[0], triplet[1], triplet[2]
            
            # genera tutti i candidati sostituendo la coda
            candidates = torch.stack([
                h.expand(num_generate),
                r.expand(num_generate),
                unique_nodes
            ], dim=1)  # (num_nodes, 3)
            
            # aggiungi il positivo vero se non è già tra i candidati
            scores = torch.sigmoid(model.distmult(embeddings, candidates))  # (num_nodes,)
            
            # rank del positivo vero
            true_score = scores[unique_nodes == t]
            if true_score.size(0) == 0:
                continue
            rank = (scores >= true_score).sum().item()  # quanti hanno score >= del positivo
            ranks_list.append(rank)
    
    if len(ranks_list) == 0:
        return 0.0, {h: 0.0 for h in hits}
    
    ranks_tensor = torch.tensor(ranks_list, dtype=torch.float)
    mrr = torch.mean(1.0 / ranks_tensor).item()
    hits_dict = {h: (ranks_tensor <= h).float().mean().item() for h in hits}
    
    return mrr, hits_dict

def evaluation_metrics_full_bidirectional(model, embeddings, all_graph_nodes, test_triplet, device, hits=[1,3,10]):
    """
    Full-entity bidirectional MRR and Hits@k evaluation for KG link prediction.

    For each test triplet (h,r,t), ranks the true entity against all entities
    in the graph in both prediction directions:
        - tail prediction: rank of t among (h,r,e) ∀ e ∈ E
        - head prediction: rank of h among (e,r,t) ∀ e ∈ E

    This corresponds to the standard full-ranking KG evaluation protocol
    (sampling-free, global ranking). Metrics are deterministic and directly
    comparable to KG benchmarks. No filtered setting is applied (raw rank).
    """

    model.eval()
    unique_nodes = all_graph_nodes.to(device)
    num_entities = unique_nodes.size(0)

    ranks = []

    with torch.no_grad():
        for i in range(test_triplet.size(0)):
            h, r, t = test_triplet[i]

            # ---------- Tail prediction: (h,r,e) ----------
            tail_candidates = torch.stack([
                h.expand(num_entities),
                r.expand(num_entities),
                unique_nodes
            ], dim=1)

            tail_scores = model.distmult(embeddings, tail_candidates)
            true_tail_score = tail_scores[unique_nodes == t]

            if true_tail_score.numel() > 0:
                tail_rank = (tail_scores >= true_tail_score).sum().item()
                ranks.append(tail_rank)

            # ---------- Head prediction: (e,r,t) ----------
            head_candidates = torch.stack([
                unique_nodes,
                r.expand(num_entities),
                t.expand(num_entities)
            ], dim=1)

            head_scores = model.distmult(embeddings, head_candidates)
            true_head_score = head_scores[unique_nodes == h]

            if true_head_score.numel() > 0:
                head_rank = (head_scores >= true_head_score).sum().item()
                ranks.append(head_rank)

    if len(ranks) == 0:
        return 0.0, {k: 0.0 for k in hits}

    ranks_tensor = torch.tensor(ranks, dtype=torch.float, device=device)

    mrr = torch.mean(1.0 / ranks_tensor).item()
    hits_dict = {k: (ranks_tensor <= k).float().mean().item() for k in hits}

    return mrr, hits_dict



########


"""
Evaluation metrics per Knowledge Graph Link Prediction — Filtered Setting.

Queste metriche sono standard nella letteratura KGE e forniscono una valutazione più realistica

Queste metriche sono full graph, non campionate, e possono essere computazionalmente intensive su grafi molto grandi. Tuttavia, sono più affidabili rispetto a metriche campionate o non filtrate, specialmente in contesti con molti positivi noti (come i KGE biomedici).

Versione corretta e migliorata rispetto alla proposta originale.
Cambiamenti rispetto alla versione proposta:
  1. Aggiunto check di sicurezza per evitare rank=0 e divisione per zero nel MRR
  2. Aggiunto clamp del rank minimo a 1 (difesa contro edge case)
  3. Separazione metriche tail/head per diagnostica
  4. Logging opzionale per debug
  5. Supporto per device mismatch
  6. Gestione edge case: nodo positivo assente da unique_nodes
"""

import torch
from collections import defaultdict


def build_positive_maps(all_target_triplets):
    """
    Costruisce le mappe dei positivi noti per il filtered setting.
    
    Args:
        all_target_triplets: tensor (N, 3) con TUTTE le triple positive 
                             (train + val + test) della relazione target.
    
    Returns:
        all_positives_tail: dict (h, r) -> set of t
        all_positives_head: dict (r, t) -> set of h
    """
    all_positives_tail = defaultdict(set)
    all_positives_head = defaultdict(set)
    
    for i in range(all_target_triplets.size(0)):
        h = all_target_triplets[i, 0].item()
        r = all_target_triplets[i, 1].item()
        t = all_target_triplets[i, 2].item()
        all_positives_tail[(h, r)].add(t)
        all_positives_head[(r, t)].add(h)
    
    return all_positives_tail, all_positives_head


def evaluation_metrics_filtered(
    model, 
    embeddings, 
    all_target_triplets,  # train + val + test della relazione target
    test_triplets,        # solo le triple di test
    all_graph_nodes,      # tutti i nodi unici del grafo
    device, 
    hits_k=[1, 3, 10],
    verbose=False
):
    """
    Calcola MRR e Hits@K con filtered setting (standard nella letteratura KGE).
    
    Il filtered setting rimuove dal ranking tutti i veri positivi noti
    (tranne quello che si sta valutando), evitando di penalizzare il modello
    per aver assegnato score alti ad altre risposte corrette.
    
    Args:
        model: modello con metodo .distmult(embeddings, triplets) -> scores
        embeddings: embedding dei nodi (output del GNN encoder)
        all_target_triplets: tensor (N, 3) — TUTTE le triple positive (train+val+test)
        test_triplets: tensor (M, 3) — solo le triple di test da valutare
        all_graph_nodes: tensor (E,) — tutti i nodi unici del grafo
        device: torch device
        hits_k: lista di K per Hits@K (default [1, 3, 10])
        verbose: se True, stampa info di debug ogni 100 triple
    
    Returns:
        dict con chiavi: 'mrr', 'mrr_tail', 'mrr_head', 
                         'hits@K' per ogni K, 'hits@K_tail', 'hits@K_head'
    """
    model.eval()
    unique_nodes = all_graph_nodes.to(device)
    num_entities = unique_nodes.size(0)
    
    # Mappa nodo -> indice nel vettore unique_nodes
    node_to_idx = {}
    for i in range(num_entities):
        node_to_idx[unique_nodes[i].item()] = i
    
    # Costruisci mappe dei positivi per il filtering
    all_positives_tail, all_positives_head = build_positive_maps(all_target_triplets)
    
    tail_ranks = []
    head_ranks = []
    skipped = 0
    
    with torch.no_grad():
        for i in range(test_triplets.size(0)):
            h, r, t = test_triplets[i]
            h_i, r_i, t_i = h.item(), r.item(), t.item()
            
            # Verifica che entrambi i nodi siano nel grafo
            if h_i not in node_to_idx or t_i not in node_to_idx:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] Tripla {i}: nodo mancante da unique_nodes")
                continue
            
            # ============================================
            # TAIL PREDICTION: dato (h, r, ?), ranking di t
            # ============================================
            tail_candidates = torch.stack([
                h.expand(num_entities).to(device),
                r.expand(num_entities).to(device),
                unique_nodes
            ], dim=1)
            tail_scores = model.distmult(embeddings, tail_candidates)
            
            # Score del vero positivo
            true_tail_score = tail_scores[node_to_idx[t_i]]
            
            # Filtered: maschera i positivi noti tranne t_i
            filter_mask = torch.ones(num_entities, dtype=torch.bool, device=device)
            for known_t in all_positives_tail.get((h_i, r_i), set()):
                if known_t != t_i and known_t in node_to_idx:
                    filter_mask[node_to_idx[known_t]] = False
            
            # Il positivo t_i NON viene mascherato (la condizione known_t != t_i lo protegge)
            # Quindi true_tail_score è incluso in filtered_scores
            filtered_scores = tail_scores[filter_mask]
            
            # Rank = quanti score sono >= al positivo (incluso se stesso, quindi rank minimo = 1)
            tail_rank = (filtered_scores >= true_tail_score).sum().item()
            
            # Safety: rank deve essere almeno 1 (difesa contro edge case numerici)
            tail_rank = max(tail_rank, 1)
            tail_ranks.append(tail_rank)
            
            # ============================================
            # HEAD PREDICTION: dato (?, r, t), ranking di h
            # ============================================
            head_candidates = torch.stack([
                unique_nodes,
                r.expand(num_entities).to(device),
                t.expand(num_entities).to(device)
            ], dim=1)
            head_scores = model.distmult(embeddings, head_candidates)
            
            true_head_score = head_scores[node_to_idx[h_i]]
            
            filter_mask = torch.ones(num_entities, dtype=torch.bool, device=device)
            for known_h in all_positives_head.get((r_i, t_i), set()):
                if known_h != h_i and known_h in node_to_idx:
                    filter_mask[node_to_idx[known_h]] = False
            
            filtered_scores = head_scores[filter_mask]
            head_rank = (filtered_scores >= true_head_score).sum().item()
            head_rank = max(head_rank, 1)
            head_ranks.append(head_rank)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Valutate {i+1}/{test_triplets.size(0)} triple "
                      f"(tail_rank={tail_rank}, head_rank={head_rank})")
    
    if skipped > 0:
        print(f"  [WARN] {skipped} triple saltate per nodi mancanti")
    
    if len(tail_ranks) == 0:
        print("  [ERROR] Nessuna tripla valutata!")
        return {
            'mrr': 0.0, 'mrr_tail': 0.0, 'mrr_head': 0.0,
            **{f'hits@{k}': 0.0 for k in hits_k},
            **{f'hits@{k}_tail': 0.0 for k in hits_k},
            **{f'hits@{k}_head': 0.0 for k in hits_k},
            'num_evaluated': 0, 'num_skipped': skipped
        }
    
    # Converti in tensori
    tail_ranks_t = torch.tensor(tail_ranks, dtype=torch.float, device=device)
    head_ranks_t = torch.tensor(head_ranks, dtype=torch.float, device=device)
    all_ranks_t = torch.cat([tail_ranks_t, head_ranks_t])
    
    # Calcola metriche
    results = {
        'mrr': (1.0 / all_ranks_t).mean().item(),
        'mrr_tail': (1.0 / tail_ranks_t).mean().item(),
        'mrr_head': (1.0 / head_ranks_t).mean().item(),
        'num_evaluated': len(tail_ranks),
        'num_skipped': skipped,
    }
    
    for k in hits_k:
        results[f'hits@{k}'] = (all_ranks_t <= k).float().mean().item()
        results[f'hits@{k}_tail'] = (tail_ranks_t <= k).float().mean().item()
        results[f'hits@{k}_head'] = (head_ranks_t <= k).float().mean().item()
    
    return results


# ============================================
# Funzione helper per stampare i risultati
# ============================================
def print_metrics(results, title="Evaluation Results"):
    """Stampa le metriche in formato leggibile."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Triple valutate: {results['num_evaluated']}"
          f" (saltate: {results['num_skipped']})")
    print(f"  MRR (overall):   {results['mrr']:.4f}")
    print(f"  MRR (tail):      {results['mrr_tail']:.4f}")
    print(f"  MRR (head):      {results['mrr_head']:.4f}")
    for key in sorted(results.keys()):
        if key.startswith('hits@') and '_' not in key:
            k = key.replace('hits@', '')
            print(f"  Hits@{k} (overall): {results[key]:.4f}")
            print(f"  Hits@{k} (tail):    {results.get(f'hits@{k}_tail', 0):.4f}")
            print(f"  Hits@{k} (head):    {results.get(f'hits@{k}_head', 0):.4f}")
    print(f"{'='*50}\n")


# ============================================
# Esempio di utilizzo
# ============================================
"""
# 1. Raccogli TUTTE le triple della relazione target (train + val + test)
all_target_triplets = torch.cat([
    train_target_triplets,
    val_target_triplets,
    test_target_triplets
], dim=0)

# 2. Raccogli tutti i nodi unici del grafo
all_graph_nodes = torch.unique(torch.cat([
    graph.edge_index[0], 
    graph.edge_index[1]
]))

# 3. Valuta
results = evaluation_metrics_filtered(
    model=model,
    embeddings=embeddings,
    all_target_triplets=all_target_triplets,
    test_triplets=test_target_triplets,
    all_graph_nodes=all_graph_nodes,
    device=device,
    hits_k=[1, 3, 10],
    verbose=True
)

print_metrics(results, title="PathogenKG — Filtered Setting")
"""