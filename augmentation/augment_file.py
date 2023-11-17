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

input_file = 'data/py150_v1.0/raw/train.txt'
output_file = 'data-aug/py150_v1.0/raw/train-aug.txt'
combined_file = 'data-aug/py150_v1.0/raw/train.txt'
process_file(input_file, output_file, combined_file)
