import os
import pdb

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
                # message if the file does not exist.
                try:
                    id_log_file = f"id_{split}_{log_type}"
                    result_dfs[id_log_file] = pd.read_csv(
                        os.path.join(log_url, f"{id_log_file}.csv")
                    )
                except:
                    print(f"{id_log_file} not available at {log_url}")

            result_dfs[f"{split}_{log_type}"] = pd.read_csv(log_csv_path)
    return result_dfs


def compile_results(dataset, results, in_distribution_val=False):
    metrics = get_metrics(dataset)
    compiled_results = dict()

    for algorithm, result_dfs in results.items():
        compiled_results[algorithm] = dict()
        print(f"\ndataset={dataset}, algorithm={algorithm}:")

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
                    sort_metrics = [metric]
                    # Use the "acc_avg" metric as a tie-breaker for datasets that have "acc_avg"
                    if metric != "acc_avg" and dataset in [
                        "amazon",
                        "camelyon17",
                        "fmow",
                        "iwildcam",
                    ]:
                        sort_metrics.append("acc_avg")

                    val_split = (
                        "id_val_eval" if in_distribution_val else "val_eval"
                    )
                    result = get_early_stopped_row(
                        result_df[result_type],
                        result_df[val_split],
                        sort_fields=sort_metrics,
                    )
                    compiled_results[algorithm][result_type][metric].append(
                        result[metric]
                    )

        for result_type in RESULT_TYPES:
            if result_type in compiled_results[algorithm]:
                for metric in metrics:
                    # Take the standard deviation and mean across replicants for each metric
                    compiled_results[algorithm][result_type][f"{metric}_std"] = np.std(
                        compiled_results[algorithm][result_type][metric]
                    )
                    compiled_results[algorithm][result_type][metric] = np.mean(
                        compiled_results[algorithm][result_type][metric]
                    )
    return compiled_results


def get_early_stopped_row(src_df, val_df, sort_fields, ascending=False):
    epoch = val_df.sort_values(sort_fields, ascending=ascending).iloc[0]["epoch"]
    print(f"Early stopped at {epoch}.")
    selected_row = src_df.loc[src_df["epoch"] == epoch]
    assert selected_row.shape[0] == 1
    return selected_row.iloc[0]


def get_metrics(dataset):
    if dataset == "amazon":
        return ["acc_avg", "10th_percentile_acc"]
    elif dataset == "camelyon17":
        return ["acc_avg"]
    elif dataset == "civilcomments":
        return ["acc_avg", "acc_wg"]
    elif dataset == "fmow":
        return ["acc_avg", "acc_worst_region"]
    elif dataset == "iwildcam":
        return ["acc_avg", "F1-macro_all"]
    elif dataset == "ogb-molpcba":
        return ["ap"]
    elif dataset == "poverty":
        return ["r_all", "r_urban:0", "r_urban:1", "r_wg"]
    else:
        raise RuntimeError(f"Fetching metrics for {dataset} is not handled.")
