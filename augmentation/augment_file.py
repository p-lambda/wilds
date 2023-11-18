import os
from augmentation.generate_refactoring import *

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

def process_file(input_file_path, output_file_path, combined_file_path, k=1):
    ensure_directory_exists(output_file_path)
    ensure_directory_exists(combined_file_path)

    with open(input_file_path, 'r') as file:
        content = file.read().split('</s>')

    new_content = []

    for snippet in content:
        if '<s>' in snippet:
            formatted_code = format_python_code(snippet)
            refactored_code = generate_adversarial_file_level(k, formatted_code)
            original_style_code = reformat_to_original_style(refactored_code)
            new_content.append(original_style_code)

    with open(output_file_path, 'w') as file:
        file.write('\n'.join(new_content))

    # Combine the original and new files into a third file
    with open(combined_file_path, 'w') as combined_file:
        with open(input_file_path, 'r') as original_file:
            combined_file.write(original_file.read() + '\n')

        with open(output_file_path, 'r') as new_file:
            combined_file.write(new_file.read())

def duplicate_meta_content(meta_file_path, output_meta_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)

    with open(meta_file_path, 'r') as meta_file:
        meta_lines = meta_file.read().splitlines()

    # Duplicate the contents
    duplicated_content = meta_lines + meta_lines

    # Write the duplicated content to the output file
    with open(output_meta_path, 'w') as output_meta:
        output_meta.write('\n'.join(duplicated_content))

# ATTENTION!!! BE EXTREMELY CAREFUL THAT THE PATHS ARE RIGHT WHEN RUNNING THIS SCRIPT
# Below are the paths for the full set and the 500 set. If you make a new set,
# you will need to add the paths. Otherwise, just comment out the ones you don't need.

######################## Augment snippet files ###############################

# Augment the full set `data/py150_v1.0/raw/train.txt`
# input_file = 'data/py150_v1.0/raw/train.txt'
# output_file = 'data-aug/py150_v1.0/raw/train-aug.txt'
# combined_file = 'data-aug/py150_v1.0/raw/train.txt'

# Augment the 500 set `data500/py150_v1.0/raw/train.txt`
input_file = 'data500/py150_v1.0/raw/train.txt'
output_file = 'data500-aug/py150_v1.0/raw/train-aug.txt'
combined_file = 'data500-aug/py150_v1.0/raw/train.txt'

process_file(input_file, output_file, combined_file)

######################## Duplicate meta files ###############################

# Duplicate the full set metadata `data/py150_v1.0/metadata/repo_file_names/train.txt`
# meta_file_path = 'data/py150_v1.0/metadata/repo_file_names/train.txt'
# output_meta_path = 'data-aug/py150_v1.0/metadata/repo_file_names/train.txt'

# Duplicate the 500 set metadata `data500/py150_v1.0/metadata/repo_file_names/train.txt`
meta_file_path = 'data500/py150_v1.0/metadata/repo_file_names/train.txt'
output_meta_path = 'data500-aug/py150_v1.0/metadata/repo_file_names/train.txt'

duplicate_meta_content(meta_file_path, output_meta_path)
