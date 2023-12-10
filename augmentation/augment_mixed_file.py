import sys
import os
import shutil
import argparse
from generate_refactoring import *

# Constants for dataset directories and augmentation settings
DATA_DIR = "data500"
AUG_DIR = "data500-aug"
SIZE_OF_DATASET = 500
APPLY_K_PER_SNIPPET = 1
SIZE_TO_AUGMENT = (SIZE_OF_DATASET / 2) * APPLY_K_PER_SNIPPET

# List of refactoring functions to be applied
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
    insert_random_function,
    insert_random_class,
    create_typo,
    dead_branch_if_else,
    dead_branch_if,
    dead_branch_while,
    dead_branch_for,
    insert_safe_random_space,
]

# Calculating the limit for refactoring based on the dataset size and number of refactorings
NO_REFACTORINGS = len(refactors_list)
RELAXATION_FACTOR = 1
PERCENT_LIMIT = (100 / NO_REFACTORINGS + RELAXATION_FACTOR) / 100
REFACTOR_LIMIT = int(PERCENT_LIMIT * SIZE_TO_AUGMENT)


def custom_print(*args, **kwargs):
    """
    Custom print function that outputs to both the terminal and a log file.

    :param args: Variable length argument list.
    :param kwargs: Arbitrary keyword arguments.
    """
    print(
        *args, **kwargs, file=original_stdout
    )  # Print to the original stdout (terminal)
    print(*args, **kwargs, file=log_file)  # Print to the log file


# Setup argparse for command-line argument parsing
parser = argparse.ArgumentParser(
    description="Run script and redirect output to specified directory"
)
parser.add_argument("output_dir", type=str, help="Directory to save the output logs")
args = parser.parse_args()

# Create the output log directory if it doesn't exist
output_log_directory = args.output_dir
os.makedirs(output_log_directory, exist_ok=True)

# Redirect stdout to a log file
original_stdout = sys.stdout  # Save original stdout
log_file_path = os.path.join(output_log_directory, "aug.log")
log_file = open(log_file_path, "w")
sys.stdout = log_file  # Set stdout to the log file


def format_python_code(snippet):
    """
    Formats a Python code snippet by replacing specific markers with newline characters.

    :param snippet: The Python code snippet to format.
    :return: The formatted code snippet.
    """
    formatted_code = snippet.replace(" <EOL>", "\n")
    formatted_code = formatted_code.replace("<s>", "").replace("</s>", "")
    return formatted_code


def reformat_to_original_style(code):
    """
    Reformats a Python code snippet to its original style by replacing newline characters with specific markers.

    :param code: The Python code snippet to reformat.
    :return: The reformatted code snippet.
    """
    formatted_code = code.replace("\n", " <EOL>")
    return f"<s> {formatted_code} </s>"


def ensure_directory_exists(file_path):
    """
    Ensures that the directory for a given file path exists. Creates the directory if it does not exist.

    :param file_path: The file path for which the directory needs to be checked/created.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_file(input_file_path, output_file_path, combined_file_path):
    """
    Processes a file by applying refactoring functions to its contents and writing the modified contents to new files.

    :param input_file_path: The path to the input file.
    :param output_file_path: The path to the file where the modified content will be saved.
    :param combined_file_path: The path to the file where combined content (original and modified) will be saved.
    """
    ensure_directory_exists(output_file_path)
    ensure_directory_exists(combined_file_path)

    with open(input_file_path, "r") as file:
        content = file.read().split("</s>")

    new_content = []
    cumulative_refactoring_counts = {
        refactor.__name__: 0 for refactor in refactors_list
    }

    midpoint = len(content) // 2
    for snippet in content[midpoint:]:
        if "<s>" in snippet:
            formatted_code = format_python_code(snippet)
            cumulative_refactoring_counts_copy = cumulative_refactoring_counts.copy()
            refactored_code, refactoring_counts = generate_adversarial_file_level(
                formatted_code,
                refactors_list,
                APPLY_K_PER_SNIPPET,
                REFACTOR_LIMIT,
                cumulative_refactoring_counts_copy,
            )
            for refactor, count in refactoring_counts.items():
                cumulative_refactoring_counts[refactor] += count
            original_style_code = reformat_to_original_style(refactored_code)
            new_content.append(original_style_code)

    for refactor, count in cumulative_refactoring_counts.items():
        custom_print(f"{refactor}: Applied {count} times")

    total = sum(cumulative_refactoring_counts.values())
    custom_print(f"Total applied: {total} times")

    with open(output_file_path, "w") as output_file:
        output_file.write("\n".join(new_content))

    with open(input_file_path, "r") as input_file:
        orig_snippets = input_file.read().splitlines()

    with open(output_file_path, "r") as new_file:
        aug_snippets = new_file.read().splitlines()

    mixed_snippets = []

    for i in range(midpoint):
        mixed_snippets.append(orig_snippets[i])

    for i in range(midpoint):
        mixed_snippets.append(aug_snippets[i])

    with open(combined_file_path, "w") as combined_file:
        combined_file.write("\n".join(mixed_snippets))


def copy_file_or_directory(source, destination):
    """
    Copy a file or directory from the source path to the destination path.
    If the destination is a directory and already exists, it will be replaced.

    :param source: The source path of the file or directory.
    :param destination: The destination path where the file or directory will be copied.
    """
    ensure_directory_exists(destination)
    if os.path.isdir(source):
        # If the destination directory exists, remove it first
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)


######################## Augment snippet files ###############################

input_file = f"{DATA_DIR}/py150_v1.0/raw/train.txt"
output_file = f"{AUG_DIR}/py150_v1.0/raw/train-aug.txt"
combined_file = f"{AUG_DIR}/py150_v1.0/raw/train.txt"

process_file(input_file, output_file, combined_file)

######################## Copy the dataset files ####################
datasets_to_copy = {
    f"{DATA_DIR}/py150_v1.0/raw/OODval.txt": f"{AUG_DIR}/py150_v1.0/raw/OODval.txt",
    f"{DATA_DIR}/py150_v1.0/raw/OODtest.txt": f"{AUG_DIR}/py150_v1.0/raw/OODtest.txt",
    f"{DATA_DIR}/py150_v1.0/raw/IDval.txt": f"{AUG_DIR}/py150_v1.0/raw/IDval.txt",
    f"{DATA_DIR}/py150_v1.0/raw/IDtest.txt": f"{AUG_DIR}/py150_v1.0/raw/IDtest.txt",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/OODval.txt": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/OODval.txt",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/OODtest.txt": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/OODtest.txt",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/IDval.txt": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/IDval.txt",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/IDtest.txt": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/IDtest.txt",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/repo_ids.csv": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/repo_ids.csv",
    f"{DATA_DIR}/py150_v1.0/metadata/repo_file_names/train.txt": f"{AUG_DIR}/py150_v1.0/metadata/repo_file_names/train.txt",
    f"{DATA_DIR}/py150_v1.0/script": f"{AUG_DIR}/py150_v1.0/script",
    f"{DATA_DIR}/py150_v1.0/RELEASE_v1.0.txt": f"{AUG_DIR}/py150_v1.0/RELEASE_v1.0.txt",
}

for input_path, output_path in datasets_to_copy.items():
    copy_file_or_directory(input_path, output_path)

# Close the log file and restore sys.stdout
log_file.close()
sys.stdout = original_stdout

print(f"Script execution completed. Logs are saved in {log_file_path}")
