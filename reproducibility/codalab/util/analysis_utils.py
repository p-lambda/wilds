import os

import numpy as np
import pandas as pd


DEFAULT_SPLITS = ["train", "val", "test"]
RESULT_TYPES = ["id_val_eval", "val_eval", "id_test_eval", "test_eval"]


def load_results(
    log_url, splits=None, include_algorithm_logs=False, include_in_distribution=False
):
    result_dfs = {}
    log_types = ["eval"]

    if not splits:
        splits = DEFAULT_SPLITS

    if include_algorithm_logs:
        log_types.append("algo")

    for split in splits:
        for log_type in log_types:
            log_csv_path = os.path.join(log_url, f"{split}_{log_type}.csv")

            if include_in_distribution and split != "train":
                # Not all datasets have a ID test and val, so just print a
                # message if the file does not exist
                try:
                    id_log_file = f"id_{split}_{log_type}"
                    result_dfs[id_log_file] = pd.read_csv(
                        os.path.join(log_url, f"{id_log_file}.csv")
                    )
                except:
                    print(f"{id_log_file} not available at {log_url}.")

            # Read the csv file and store in the results dataframe
            result_dfs[f"{split}_{log_type}"] = pd.read_csv(log_csv_path)
    return result_dfs


def compile_results(dataset, results, in_distribution_val=False):
    metrics = get_metrics(dataset)
    print(f"Early stopping with {metrics}...")

    compiled_results = dict()
    for algorithm, result_dfs in results.items():
        compiled_results[algorithm] = dict()
        print(f"\nDataset={dataset}, Algorithm={algorithm}")

        for result_type in RESULT_TYPES:
            if len(result_dfs) > 0 and result_type in result_dfs[0]:
                compiled_results[algorithm][result_type] = dict()
                for metric in metrics:
                    compiled_results[algorithm][result_type][metric] = []

        for result_df in result_dfs:
            for result_type in RESULT_TYPES:
                if result_type not in result_df:
                    continue

                for metric in metrics:
                    val_split = "id_val_eval" if in_distribution_val else "val_eval"
                    result = get_early_stopped_row(
                        result_df[result_type],
                        result_df[val_split],
                        sort_metrics=metrics,
                    )
                    compiled_results[algorithm][result_type][metric].append(
                        result[metric]
                    )

        for result_type in RESULT_TYPES:
            if result_type in compiled_results[algorithm]:
                for metric in metrics:
                    # Take the sample standard deviation (delta degree of freedom = 1)
                    # and mean across replicates for each metric.
                    compiled_results[algorithm][result_type][f"{metric}_std"] = np.std(
                        compiled_results[algorithm][result_type][metric], ddof=1
                    )
                    compiled_results[algorithm][result_type][metric] = np.mean(
                        compiled_results[algorithm][result_type][metric]
                    )
    return compiled_results


def get_early_stopped_row(src_df, val_df, sort_metrics):
    """
    Get the early stopped row with the highest metric value.
    """
    # Sort in descending order so larger metric value the better
    epoch = val_df.sort_values(sort_metrics, ascending=False).iloc[0]["epoch"]
    selected_row = src_df.loc[src_df["epoch"] == epoch]
    assert selected_row.shape[0] == 1
    return selected_row.iloc[0]


def get_metrics(dataset):
    """
    Returns a list of metrics for the given dataset. The first metric of the list
    is used to determine the best hyperparameters on a validation set.

    Parameters:
        dataset (str): Dataset name or a string containing the dataset name.
    """
    if "amazon" in dataset:
        return ["10th_percentile_acc", "acc_avg"]
    elif "camelyon17" in dataset:
        return ["acc_avg"]
    elif "civilcomments" in dataset:
        return ["acc_wg", "acc_avg"]
    elif "fmow" in dataset:
        return ["acc_worst_region", "acc_avg"]
    elif "iwildcam" in dataset:
        return ["F1-macro_all", "acc_avg"]
    elif "ogb-molpcba" in dataset:
        return ["ap"]
    elif "poverty" in dataset:
        return ["r_wg", "r_all", "r_urban:0", "r_urban:1"]
    elif "py150" in dataset:
        return ["acc"]
    else:
        raise ValueError(f"Fetching metrics for {dataset} is not handled.")
