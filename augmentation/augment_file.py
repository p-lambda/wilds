"""
Takes version of py150 dataset and outputs an augmented dataset that 
contains either an original code snippet or its augmented counterpart,
but not both.

Expects output log directory as command-line argument, to output stats
on which augmentations were applied.
"""
import sys
import os
import shutil
from generate_refactoring import *
import argparse

DATA_DIR = 'data'
AUG_DIR = 'data-aug'

# Custom print function to output both to stdout and log file
def custom_print(*args, **kwargs):
    # Print to the original stdout (terminal)
    print(*args, **kwargs, file=original_stdout)

    # Print to the log file
    print(*args, **kwargs, file=log_file)

# Parsing command line argument for the output directory
parser = argparse.ArgumentParser(description="Run script and redirect output to specified directory")
parser.add_argument('output_dir', type=str, help='Directory to save the output logs')
args = parser.parse_args()

# Create the directory if it does not exist
output_log_directory = args.output_dir
os.makedirs(output_log_directory, exist_ok=True)

# Redirect stdout to a file in the specified directory
original_stdout = sys.stdout  # Save a reference to the original standard output
log_file_path = os.path.join(output_log_directory, 'aug.log')
log_file = open(log_file_path, 'w')
sys.stdout = log_file  # Change the standard output to the file we created

def format_python_code(snippet):
    formatted_code = snippet.replace(" <EOL>", "\n")
    formatted_code = formatted_code.replace("<s>", "").replace("</s>", "")
    return formatted_code

def reformat_to_original_style(code):
    formatted_code = code.replace("\n", " <EOL>")
    return f"<s> {formatted_code} </s>"

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_file(input_file_path, output_file_path, combined_file_path, k):
    ensure_directory_exists(output_file_path)
    ensure_directory_exists(combined_file_path)

    with open(input_file_path, 'r') as file:
        content = file.read().split('</s>')

    new_content = []

    refactors_list = [
                    rename_argument, 
                    return_optimal, 
                    add_argumemts,
                    rename_api, 
                    rename_local_variable,
                    add_local_variable,
                    rename_method_name,
                    enhance_if,
                    add_print,
                    duplication,
                    apply_plus_zero_math,
                    dead_branch_if_else,
                    dead_branch_if,
                    dead_branch_while,
                    dead_branch_for,
                    ]

    cumulative_refactoring_counts = {refactor.__name__: 0 for refactor in refactors_list}

    for snippet in content:
        if '<s>' in snippet:
            formatted_code = format_python_code(snippet)
            refactored_code, refactoring_counts = generate_adversarial_file_level(formatted_code, k)
            for refactor, count in refactoring_counts.items():
                cumulative_refactoring_counts[refactor] += count
            original_style_code = reformat_to_original_style(refactored_code)
            new_content.append(original_style_code)

    # Print cumulative refactoring counts after the loop
    for refactor, count in cumulative_refactoring_counts.items():
        custom_print(f"{refactor}: Applied {count} times")

    # Write the new content to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write('\n'.join(new_content))

    # Combine mixture of original and new files into a third file
    with open(input_file_path, 'r') as input_file:
        orig_snippets = input_file.read().splitlines()
    
    with open(output_file_path, 'r') as new_file:
        aug_snippets = new_file.read().splitlines()

    mixed_snippets = []
    for i in range(0, len(orig_snippets)):
        mixed_snippets.append(orig_snippets[i] if i % 2 == 0 else aug_snippets[i])

    with open(combined_file_path, 'w') as combined_file:
        combined_file.write('\n'.join(mixed_snippets))

def copy_file_or_directory(source, destination):
    """
    Copy a file or directory from the source path to the destination path.
    If the destination is a directory and already exists, it will be replaced.
    """
    ensure_directory_exists(destination)
    if os.path.isdir(source):
        # If the destination directory exists, remove it first
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)

# ATTENTION!!! BE EXTREMELY CAREFUL THAT THE PATHS ARE RIGHT WHEN RUNNING THIS SCRIPT

######################## Augment snippet files ###############################

input_file = f'{DATA_DIR}/py150_v1.0/raw/train.txt'
output_file = f'{AUG_DIR}/py150_v1.0/raw/train-aug.txt'
combined_file = f'{AUG_DIR}/py150_v1.0/raw/train.txt'

process_file(input_file, output_file, combined_file, 1)

######################## Copy the dataset files ####################
datasets_to_copy = {
    f'{DATA_DIR}/py150_v1.0/raw/OODval.txt': f'{AUG_DIR}/py150_v1.0/raw/OODval.txt',
    f'{DATA_DIR}/py150_v1.0/raw/OODtest.txt': f'{AUG_DIR}/py150_v1.0/raw/OODtest.txt',
    f'{DATA_DIR}/py150_v1.0/raw/IDval.txt': f'{AUG_DIR}/py150_v1.0/raw/IDval.txt',
    f'{DATA_DIR}/py150_v1.0/raw/IDtest.txt': f'{AUG_DIR}/py150_v1.0/raw/IDtest.txt',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/OODval.txt': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/OODval.txt',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/OODtest.txt': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/OODtest.txt',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/IDval.txt': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/IDval.txt',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/IDtest.txt': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/IDtest.txt',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/repo_ids.csv': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/repo_ids.csv',
    f'{DATA_DIR}/py150_v1.0/metadata/repo_file_names/train.txt': f'{AUG_DIR}/py150_v1.0/metadata/repo_file_names/train.txt',
    f'{DATA_DIR}/py150_v1.0/script': f'{AUG_DIR}/py150_v1.0/script',
    f'{DATA_DIR}/py150_v1.0/RELEASE_v1.0.txt': f'{AUG_DIR}/py150_v1.0/RELEASE_v1.0.txt'
}

for input_path, output_path in datasets_to_copy.items():
    copy_file_or_directory(input_path, output_path)

# Close the log file and restore sys.stdout before the final print
log_file.close()
sys.stdout = original_stdout

# Use standard print for the final message
print(f"Script execution completed. Logs are saved in {log_file_path}")