import argparse
import datetime
import json
import os
import subprocess

import pandas as pd

from examples.configs.algorithm import algorithm_defaults
from examples.evaluate import get_metrics
from wilds import benchmark_datasets

"""
Runs the smoke tests for WILDS in CodaLab:
    1. Run ERM on all the benchmarks
    2. Run reweighted (ERM + uniform_over_groups), IRM, CORAL and GroupDRO on CivilComments

Only small fractions of the datasets are used for testing purposes. This test suite should
take around 2 hours to finish.

Usage:

    python3 tests/smoke_test.py --wilds-src-uuid <UUID of the WILDS source to test>
    python3 tests/smoke_test.py --run-id <ID from running the previous command>
"""


class WILDSSmokeTestSuite:
    # Only use 10% of the data splits for testing purposes
    DATASET_FRACTION = 0.1

    # Hardcoded to the UUID of the worksheet that hosts the smoke test run bundles
    TEST_WORKSHEET_UUID = "0xacd40b3e4991410b98643b9cc0b10347"

    def __init__(self):
        # Switch to the test worksheet
        self._run_bash_command(["cl", "work", WILDSSmokeTestSuite.TEST_WORKSHEET_UUID])

    def run(self, wilds_src_uuid):
        def run_experiment(
            experiment_name, benchmark, algorithm, uniform_over_groups=False
        ):
            # Run with default seed and hyperparameters
            run_expt_command = (
                "python wilds/examples/run_expt.py --download --root_dir $HOME/data --log_dir $HOME "
                f"--dataset {benchmark} --algorithm {algorithm} --frac {WILDSSmokeTestSuite.DATASET_FRACTION}"
            )
            if uniform_over_groups:
                run_expt_command += " --groupby_fields y --uniform_over_groups=True"

            self._run_bash_command(
                [
                    "cl",
                    "run",
                    f"--name={experiment_name}",
                    "--request-docker-image=pangwei/wilds_src:1.0",
                    "--exclude-patterns=data",
                    "--request-cpus=4",
                    "--request-gpus=1",
                    "--request-memory=32g",
                    "--request-network",
                    f"wilds:{wilds_src_uuid}",
                    run_expt_command,
                ]
            )

        # Use the current datetime as a unique identifier
        current_datetime = datetime.datetime.now().replace(microsecond=0)
        self._add_header_to_worksheet(current_datetime)
        run_id = str(current_datetime).replace(" ", "-").replace(":", "-")

        # Test ERM on all benchmarks
        algorithm = "ERM"
        for benchmark in benchmark_datasets:
            experiment_name = f"{benchmark}_{algorithm}_{run_id}"
            run_experiment(experiment_name, benchmark, algorithm)

        # Test the rest of the algorithms CivilComments
        benchmark = "civilcomments"
        for algorithm in algorithm_defaults.keys():
            experiment_name = f"{benchmark}_{'reweighted' if algorithm == 'ERM' else algorithm}_{run_id}"
            run_experiment(
                experiment_name,
                benchmark,
                algorithm,
                uniform_over_groups=algorithm == "ERM",
            )

        print(f"\n\nUnique ID for this run:\n{run_id}")

    def evaluate(self, run_id):
        def search_finished_run(experiment_name):
            uuids = self._run_bash_command(
                [
                    "cl",
                    "search",
                    experiment_name,
                    "state=ready",
                    f"host_worksheet={WILDSSmokeTestSuite.TEST_WORKSHEET_UUID}",
                    ".limit=1",
                    "--uuid-only",
                ]
            ).split("\n")

            # Should only get back a single UUID
            if len(uuids) != 1 or not uuids[0]:
                raise RuntimeError(
                    f"Did not find a finished run for {experiment_name}."
                )
            return uuids[0]

        print(f"Evaluating {run_id}...")
        all_results = dict()

        # Gather results for the ERM runs
        algorithm = "ERM"
        for benchmark in benchmark_datasets:
            experiment_name = f"{benchmark}_{algorithm}_{run_id}"
            uuid = search_finished_run(experiment_name)
            all_results[experiment_name] = self.load_results_from_bundle(uuid)

        # Gather the rest of the results
        benchmark = "civilcomments"
        for algorithm in algorithm_defaults.keys():
            experiment_name = f"{benchmark}_{'reweighted' if algorithm == 'ERM' else algorithm}_{run_id}"
            uuid = search_finished_run(experiment_name)
            all_results[experiment_name] = self.load_results_from_bundle(uuid)

        final_results = self.compile_metrics(all_results)
        with open(
            f"tests/results_{run_id}.json",
            "w",
        ) as f:
            json.dump(final_results, f, indent=4)

    def compile_metrics(self, results):
        def get_early_stopped_row(src_df, val_df, sort_metrics):
            # Sort in descending order so larger the metric value the better
            epoch = val_df.sort_values(sort_metrics, ascending=False).iloc[0]["epoch"]
            selected_row = src_df.loc[src_df["epoch"] == epoch]
            assert selected_row.shape[0] == 1
            return selected_row.iloc[0]

        compiled_results = dict()
        for experiment_name, result_dfs in results.items():
            dataset_name = experiment_name.split("_")[0]
            metrics = get_metrics(dataset_name)
            compiled_results[experiment_name] = dict()

            for split, result_df in result_dfs.items():
                for metric in metrics:
                    result = get_early_stopped_row(
                        result_df,
                        result_dfs["val_eval"],
                        sort_metrics=metrics,
                    )
                    if split not in compiled_results[experiment_name]:
                        compiled_results[experiment_name][split] = dict()
                    compiled_results[experiment_name][split][metric] = result[metric]
        return compiled_results

    def load_results_from_bundle(self, experiment_uuid):
        log_url = f"https://worksheets.codalab.org/rest/bundles/{experiment_uuid}/contents/blob"
        result_dfs = dict()
        for split in ["val", "test"]:
            log_csv_path = os.path.join(log_url, f"{split}_eval.csv")

            # Not all datasets have a ID test and val, so just print a message if the file does not exist
            id_log_file = f"id_{split}_eval"
            try:
                result_dfs[id_log_file] = pd.read_csv(
                    os.path.join(log_url, f"{id_log_file}.csv")
                )
            except:
                print(f"{id_log_file} not available at {log_url}.")

            # Read the csv file and store in the results dataframe
            result_dfs[f"{split}_eval"] = pd.read_csv(log_csv_path)
        return result_dfs

    def _add_header_to_worksheet(self, header):
        def add_text(text):
            self._run_bash_command(["cl", "add", "text", text])

        add_text("")
        add_text("{} {}".format("#" * 2, header))
        add_text("")

    def _run_bash_command(self, command, print_output=True):
        def clean_output(output):
            # Clean output to get UUID, states, etc.
            return output.strip("\n").strip()

        print(" ".join(command))
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if print_output:
            print(process.stdout)
        return clean_output(process.stdout)


def main():
    print(args)

    test_suite = WILDSSmokeTestSuite()
    if args.run_id:
        test_suite.evaluate(args.run_id)
    else:
        if not args.wilds_src_uuid:
            print("Please specify a UUID of the WILDS source to test.")
            return

        test_suite.run(args.wilds_src_uuid)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the smoke tests for WILDS in CodaLab."
    )
    parser.add_argument(
        "--wilds-src-uuid",
        type=str,
        help="UUID of the WILDS source to test.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Evaluate the set of experiments for the given run id.",
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
