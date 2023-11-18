import random
import os

def sample_snippets(raw_file_path, meta_file_path, output_raw_path, output_meta_path, sample_size=500):
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)

    with open(raw_file_path, 'r') as raw_file:
        raw_lines = raw_file.readlines()

    total_lines = len(raw_lines)
    
    # Generate random unique line numbers
    sampled_indices = random.sample(range(total_lines), sample_size)

    # Read metadata file and extract corresponding lines
    with open(meta_file_path, 'r') as meta_file:
        meta_lines = meta_file.readlines()
    sampled_meta_lines = [meta_lines[i].rstrip('\n') for i in sampled_indices]

    # Extract the corresponding snippets
    sampled_snippets = [raw_lines[i].rstrip('\n') for i in sampled_indices]

    # Write the sampled snippets and metadata to new files
    with open(output_raw_path, 'w') as output_raw:
        output_raw.write('\n'.join(sampled_snippets))

    with open(output_meta_path, 'w') as output_meta:
        output_meta.write('\n'.join(sampled_meta_lines))

# Define file paths
raw_file_path = 'data/py150_v1.0/raw/train.txt'
meta_file_path = 'data/py150_v1.0/metadata/repo_file_names/train.txt'
output_raw_path = 'data500/py150_v1.0/raw/train.txt'
output_meta_path = 'data500/py150_v1.0/metadata/repo_file_names/train.txt'

# Sample the snippets
sample_snippets(raw_file_path, meta_file_path, output_raw_path, output_meta_path)