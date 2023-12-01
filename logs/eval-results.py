import re
from pprint import pprint

DIRECTORY = 'logs24k'

# Input directories
INPUT1 = f'{DIRECTORY}/all-1'
INPUT2 = f'{DIRECTORY}/all-all'

def parse_input(input_text):
    # Regular expressions for extracting data
    epoch_pattern = r'Epoch \[\d+\]:'
    train_pattern = r'Train:\s*objective: ([\d\.]+)\s*loss_all: ([\d\.]+)\s*acc_all: ([\d\.]+)'
    val_pattern = r'Validation \(OOD\):\s*objective: ([\d\.]+)\s*loss_all: ([\d\.]+)\s*acc_all: ([\d\.]+)'
    val_acc_pattern = r'Validation acc: ([\d\.]+)'
    detailed_acc_pattern = r'Acc \((\w+)\): ([\d\.]+)'

    data = {
        'epochs': [],
    }

    epochs = re.split(epoch_pattern, input_text)
    for epoch_data in epochs[1:]:
        train_matches = re.findall(train_pattern, epoch_data)
        val_matches = re.findall(val_pattern, epoch_data)
        val_acc_match = re.search(val_acc_pattern, epoch_data)
        detailed_acc_matches = re.findall(detailed_acc_pattern, epoch_data)

        epoch_info = {
            'train': {
                'objective': [float(x[0]) for x in train_matches],
                'loss_all': [float(x[1]) for x in train_matches],
                'acc_all': [float(x[2]) for x in train_matches],
            },
            'validation': {
                'objective': float(val_matches[0][0]) if val_matches else None,
                'loss_all': float(val_matches[0][1]) if val_matches else None,
                'acc_all': float(val_matches[0][2]) if val_matches else None,
            },
            'validation_acc': float(val_acc_match.group(1)) if val_acc_match else None,
            'detailed_acc': {match[0]: float(match[1]) for match in detailed_acc_matches}
        }
        data['epochs'].append(epoch_info)

    return data

# Function to read data from a file
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Function to extract the relevant part of the directory name
def format_label(label):
    return label.split('/')[-1]

def compare_data(data1, data2, label1, label2):
    comparison = {}
    overall_performance = {}
    num_epochs = min(len(data1['epochs']), len(data2['epochs']))

    formatted_label1 = format_label(label1)
    formatted_label2 = format_label(label2)

    for i in range(num_epochs):
        epoch_comp = {}
        better_count = {formatted_label1: 0, formatted_label2: 0, 'Equal': 0}

        # Compare Train Accuracy
        train_acc1, train_acc2 = data1['epochs'][i]['train']['acc_all'][-1], data2['epochs'][i]['train']['acc_all'][-1]
        epoch_comp['Train Accuracy'] = (train_acc1, train_acc2, better_metric(train_acc1, train_acc2, formatted_label1, formatted_label2, better_count))

        # Compare Validation Accuracy
        val_acc1, val_acc2 = data1['epochs'][i]['validation']['acc_all'], data2['epochs'][i]['validation']['acc_all']
        epoch_comp['Validation Accuracy'] = (val_acc1, val_acc2, better_metric(val_acc1, val_acc2, formatted_label1, formatted_label2, better_count))

        # Compare Validation Acc Metric
        val_acc_metric1, val_acc_metric2 = data1['epochs'][i]['validation_acc'], data2['epochs'][i]['validation_acc']
        epoch_comp['Validation Acc Metric'] = (val_acc_metric1, val_acc_metric2, better_metric(val_acc_metric1, val_acc_metric2, formatted_label1, formatted_label2, better_count))

        # Compare detailed accuracies
        for key in data1['epochs'][i]['detailed_acc'].keys():
            acc1, acc2 = data1['epochs'][i]['detailed_acc'][key], data2['epochs'][i]['detailed_acc'][key]
            epoch_comp[f'Acc ({key})'] = (acc1, acc2, better_metric(acc1, acc2, formatted_label1, formatted_label2, better_count))

        comparison[f'Epoch {i}'] = epoch_comp
        overall_performance[f'Epoch {i}'] = determine_overall_performance(better_count, formatted_label1, formatted_label2)

    return comparison, overall_performance

def better_metric(metric1, metric2, label1, label2, better_count):
    if metric1 > metric2:
        better_count[label1] += 1
        return label1
    elif metric1 < metric2:
        better_count[label2] += 1
        return label2
    else:
        better_count['Equal'] += 1
        return 'Equal'

def determine_overall_performance(better_count, label1, label2):
    if better_count[label1] > better_count[label2]:
        winner = label1
    elif better_count[label1] < better_count[label2]:
        winner = label2
    else:
        winner = 'Equal'
    
    return {
        'winner': winner,
        'details': {
            'better': better_count[winner] if winner != 'Equal' else None,
            'equal': better_count['Equal'],
            'worse': better_count[label2] if winner == label1 else better_count[label1] if winner == label2 else None
        }
    }

# Read and parse the inputs
input1_data = parse_input(read_data_from_file(f'{INPUT1}/log.txt'))
input2_data = parse_input(read_data_from_file(f'{INPUT2}/log.txt'))

# Comparing the data
comparison_result, overall_performance = compare_data(input1_data, input2_data, INPUT1, INPUT2)
pprint(comparison_result)
print("\nOverall better performance per epoch:")
pprint(overall_performance)