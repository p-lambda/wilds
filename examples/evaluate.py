import argparse
import json
import os
import urllib.request
from ast import literal_eval
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import torch

from wilds import benchmark_datasets
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset


"""
Evaluate predictions for WILDS datasets.

Usage:

    python examples/evaluate.py <Path to directory with predictions>  <Path to output directory>
    python examples/evaluate.py <Path to directory with predictions>  <Path to output directory> --dataset <A WILDS dataset>

"""


def evaluate_all_benchmarks(predictions_dir: str, output_dir: str, root_dir: str):
    """
    Evaluate predictions for all the WILDS benchmarks.

    Parameters:
        predictions_dir (str): Path to the directory with predictions. Can be a URL
        output_dir (str): Output directory
        root_dir (str): The directory where datasets can be found
    """
    all_results: Dict[str, Dict[str, Dict[str, float]]] = dict()
    for dataset in benchmark_datasets:
        try:
            all_results[dataset] = evaluate_benchmark(
                dataset, os.path.join(predictions_dir, dataset), output_dir, root_dir
            )
        except Exception as e:
            print(f"Could not evaluate predictions for {dataset}:\n{str(e)}")

    # Write out aggregated results to output file
    print(f"Writing complete results to {output_dir}...")
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)


def evaluate_benchmark(
    dataset_name: str, predictions_dir: str, output_dir: str, root_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate across multiple replicates for a single benchmark.

    Parameters:
        dataset_name (str): Name of the dataset. See datasets.py for the complete list of datasets.
        predictions_dir (str): Path to the directory with predictions. Can be a URL.
        output_dir (str): Output directory
        root_dir (str): The directory where datasets can be found

    Returns:
        Metrics as a dictionary with metrics as the keys and metric values as the values
    """

    def get_replicates(dataset_name: str) -> List[str]:
        if dataset_name == "poverty":
            return [f"fold:{fold}" for fold in ["A", "B", "C", "D", "E"]]
        else:
            if dataset_name == "camelyon17":
                seeds = range(0, 10)
            elif dataset_name == "civilcomments":
                seeds = range(0, 5)
            else:
                seeds = range(0, 3)
            return [f"seed:{seed}" for seed in seeds]

    def get_prediction_file(
        predictions_dir: str, dataset_name: str, split: str, replicate: str
    ) -> str:
        run_id = f"{dataset_name}_split:{split}_{replicate}"
        for file in os.listdir(predictions_dir):
            if file.startswith(run_id) and (
                file.endswith(".csv") or file.endswith(".pth")
            ):
                return file
        raise FileNotFoundError(
            f"Could not find CSV or pth prediction file that starts with {run_id}."
        )

    def get_metrics(dataset_name: str) -> List[str]:
        if "amazon" == dataset_name:
            return ["10th_percentile_acc", "acc_avg"]
        elif "camelyon17" == dataset_name:
            return ["acc_avg"]
        elif "civilcomments" == dataset_name:
            return ["acc_wg", "acc_avg"]
        elif "fmow" == dataset_name:
            return ["acc_worst_region", "acc_avg"]
        elif "iwildcam" == dataset_name:
            return ["F1-macro_all", "acc_avg"]
        elif "ogb-molpcba" == dataset_name:
            return ["ap"]
        elif "poverty" == dataset_name:
            return ["r_wg", "r_all"]
        elif "py150" == dataset_name:
            return ["acc", "Acc (Overall)"]
        elif "globalwheat" == dataset_name:
            return ["detection_acc_avg_dom"]
        elif "rxrx1" == dataset_name:
            return ["acc_avg"]
        else:
            raise ValueError(f"Invalid dataset: {dataset_name}")

    # Dataset will only be downloaded if it does not exist
    wilds_dataset: WILDSDataset = get_dataset(
        dataset=dataset_name, root_dir=root_dir, download=True
    )
    splits: List[str] = list(wilds_dataset.split_dict.keys())
    if "train" in splits:
        splits.remove("train")

    replicates_results: Dict[str, Dict[str, List[float]]] = dict()
    replicates: List[str] = get_replicates(dataset_name)
    metrics: List[str] = get_metrics(dataset_name)

    # Store the results for each replicate
    for split in splits:
        replicates_results[split] = {}
        for metric in metrics:
            replicates_results[split][metric] = []

        for replicate in replicates:
            predictions_file = get_prediction_file(
                predictions_dir, dataset_name, split, replicate
            )
            print(
                f"Processing split={split}, replicate={replicate}, predictions_file={predictions_file}..."
            )
            full_path = os.path.join(predictions_dir, predictions_file)

            # GlobalWheat's predictions are a list of dictionaries, so it has to be handled separately
            if dataset_name == "globalwheat":
                metric_results: Dict[str, float] = evaluate_replicate_for_globalwheat(
                    wilds_dataset, split, full_path
                )
            else:
                predicted_labels: torch.Tensor = get_predictions(full_path)
                metric_results = evaluate_replicate(
                    wilds_dataset, split, predicted_labels
                )
            for metric in metrics:
                replicates_results[split][metric].append(metric_results[metric])

    aggregated_results: Dict[str, Dict[str, float]] = dict()

    # Aggregate results of replicates
    for split in splits:
        aggregated_results[split] = {}
        for metric in metrics:
            replicates_metric_values: List[float] = replicates_results[split][metric]
            aggregated_results[split][f"{metric}_std"] = np.std(
                replicates_metric_values, ddof=1
            )
            aggregated_results[split][metric] = np.mean(replicates_metric_values)

    # Write out aggregated results to output file
    print(f"Writing aggregated results for {dataset_name} to {output_dir}...")
    with open(os.path.join(output_dir, f"{dataset_name}_results.json"), "w") as f:
        json.dump(aggregated_results, f, indent=4)

    return aggregated_results


def evaluate_replicate(
    dataset: WILDSDataset, split: str, predicted_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate the given predictions and return the appropriate metrics.

    Parameters:
        dataset (WILDSDataset): A WILDS Dataset
        split (str): split we are evaluating on
        predicted_labels (torch.Tensor): Predictions

    Returns:
        Metrics as a dictionary with metrics as the keys and metric values as the values
    """
    # Dataset will only be downloaded if it does not exist
    subset: WILDSSubset = dataset.get_subset(split)
    metadata: torch.Tensor = subset.metadata_array
    true_labels = subset.y_array
    if predicted_labels.shape != true_labels.shape:
        predicted_labels.unsqueeze_(-1)
    return dataset.eval(predicted_labels, true_labels, metadata)[0]


def evaluate_replicate_for_globalwheat(
    dataset: WILDSDataset, split: str, path_to_predictions: str
) -> Dict[str, float]:
    predicted_labels = torch.load(path_to_predictions)
    subset: WILDSSubset = dataset.get_subset(split)
    metadata: torch.Tensor = subset.metadata_array
    true_labels = [subset.dataset.y_array[idx] for idx in subset.indices]
    return dataset.eval(predicted_labels, true_labels, metadata)[0]


def get_predictions(path: str) -> torch.Tensor:
    """
    Extract out the predictions from the file at path.

    Parameters:
        path (str): Path to the file that has the predicted labels. Can be a URL.

    Return:
        Tensor representing predictions
    """
    if is_path_url(path):
        data = urllib.request.urlopen(path)
    else:
        file = open(path, mode="r")
        data = file.readlines()
        file.close()

    predicted_labels = [literal_eval(line.rstrip()) for line in data if line.rstrip()]
    return torch.from_numpy(np.array(predicted_labels))


def is_path_url(path: str) -> bool:
    """
    Returns True if the path is a URL.
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def main():
    if args.dataset:
        evaluate_benchmark(
            args.dataset, args.predictions_dir, args.output_dir, args.root_dir
        )
    else:
        print("A dataset was not specified. Evaluating for all WILDS datasets...")
        evaluate_all_benchmarks(args.predictions_dir, args.output_dir, args.root_dir)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for WILDS datasets."
    )
    parser.add_argument(
        "predictions_dir",
        type=str,
        help="Path to prediction CSV or pth files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=benchmark_datasets,
        help="WILDS dataset to evaluate for.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data",
        help="The directory where the datasets can be found (or should be downloaded to, if they do not exist).",
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
