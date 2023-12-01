import random
import os
import shutil

INPUT = 'data'
OUTPUT = 'data24k'

def calculate_sample_sizes(train_sample_size, proportions):
    """
    Calculate sample sizes for other datasets based on the proportion and training dataset sample size.
    """
    total = sum(proportions.values())
    return {key: round((value / total) * train_sample_size) for key, value in proportions.items()}

def sample_from_file(input_path, output_path, sample_size, sample_indices=None):
    """
    Sample lines from a file based on sample_indices or a given sample_size.
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()

    if sample_indices is None:
        sample_indices = random.sample(range(len(lines)), sample_size)

    sampled_lines = [lines[i].rstrip('\n') for i in sample_indices]

    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join(sampled_lines))

    return sample_indices

def sample_snippets(raw_file_path, meta_file_path, output_raw_path, output_meta_path, sample_size, other_datasets_paths=None, other_metadata_paths=None):
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)

    # Sample from the training dataset
    train_sample_indices = sample_from_file(raw_file_path, output_raw_path, sample_size)
    sample_from_file(meta_file_path, output_meta_path, sample_size, sample_indices=train_sample_indices)

    # Sample from other datasets
    if other_datasets_paths and other_metadata_paths:
        for dataset, paths in other_datasets_paths.items():
            input_raw_path, output_raw_path = paths
            input_meta_path, output_meta_path = other_metadata_paths[dataset]
            dataset_sample_size = calculate_sample_sizes(sample_size, proportions)[dataset]
            
            # Sample raw data and metadata for other datasets
            dataset_sample_indices = sample_from_file(input_raw_path, output_raw_path, dataset_sample_size)
            sample_from_file(input_meta_path, output_meta_path, dataset_sample_size, sample_indices=dataset_sample_indices)
    
    # Check and copy additional files
    repo_ids_dest = f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/repo_ids.csv'
    if os.path.exists(repo_ids_dest):
        os.remove(repo_ids_dest)
    shutil.copy(f'{INPUT}/py150_v1.0/metadata/repo_file_names/repo_ids.csv', repo_ids_dest)

    script_dest = f'{OUTPUT}/py150_v1.0/script'
    if os.path.exists(script_dest):
        shutil.rmtree(script_dest)
    shutil.copytree(f'{INPUT}/py150_v1.0/script', script_dest)

    release_file_dest = f'{OUTPUT}/py150_v1.0/RELEASE_v1.0.txt'
    if os.path.exists(release_file_dest):
        os.remove(release_file_dest)
    shutil.copy(f'{INPUT}/py150_v1.0/RELEASE_v1.0.txt', release_file_dest)

# Define file paths and proportions
proportions = {
    'OODval': 3.44,
    'OODtest': 26.65,
    'IDval': 3.33,
    'IDtest': 13.33
}

other_datasets_paths = {
    'OODval': (f'{INPUT}/py150_v1.0/raw/OODval.txt', f'{OUTPUT}/py150_v1.0/raw/OODval.txt'),
    'OODtest': (f'{INPUT}/py150_v1.0/raw/OODtest.txt', f'{OUTPUT}/py150_v1.0/raw/OODtest.txt'),
    'IDval': (f'{INPUT}/py150_v1.0/raw/IDval.txt', f'{OUTPUT}/py150_v1.0/raw/IDval.txt'),
    'IDtest': (f'{INPUT}/py150_v1.0/raw/IDtest.txt', f'{OUTPUT}/py150_v1.0/raw/IDtest.txt')
}

other_metadata_paths = {
    'OODval': (f'{INPUT}/py150_v1.0/metadata/repo_file_names/OODval.txt', f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/OODval.txt'),
    'OODtest': (f'{INPUT}/py150_v1.0/metadata/repo_file_names/OODtest.txt', f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/OODtest.txt'),
    'IDval': (f'{INPUT}/py150_v1.0/metadata/repo_file_names/IDval.txt', f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/IDval.txt'),
    'IDtest': (f'{INPUT}/py150_v1.0/metadata/repo_file_names/IDtest.txt', f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/IDtest.txt')
}

# Sample the snippets
sample_snippets(
    raw_file_path=f'{INPUT}/py150_v1.0/raw/train.txt',
    meta_file_path=f'{INPUT}/py150_v1.0/metadata/repo_file_names/train.txt',
    output_raw_path=f'{OUTPUT}/py150_v1.0/raw/train.txt',
    output_meta_path=f'{OUTPUT}/py150_v1.0/metadata/repo_file_names/train.txt',
    sample_size=24000,
    other_datasets_paths=other_datasets_paths,
    other_metadata_paths=other_metadata_paths
)
