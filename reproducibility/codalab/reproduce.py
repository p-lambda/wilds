import argparse
import json
import math
import pdb
import subprocess
from itertools import product

import numpy as np

from examples.configs.algorithm import algorithm_defaults
from examples.configs.datasets import dataset_defaults
from reproducibility.codalab.hyperparameter_search_space import (
    CORAL_HYPERPARAMETER_SEARCH_SPACE,
    DANN_HYPERPARAMETER_SEARCH_SPACE,
)
from reproducibility.codalab.util.analysis_utils import (
    compile_results,
    get_early_stopped_row,
    get_metrics,
    load_results,
)

# Fix the seed for reproducibility
np.random.seed(0)

"""
Reproduce results of the WILDS Unlabeled paper in CodaLab.

Usage:
    python reproducibility/codalab/reproduce.py <args>
    
Example Usage:
    # To tune model hyperparameters for Unlabeled WILDS
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm deepCORAL
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_deepcoral_tune
  
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm deepCORAL --coarse
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_deepcoral_coarse_tune

    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm DANN --random
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_dann_tune
    
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm DANN --coarse --random
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_dann_coarse_tune
    
    python reproducibility/codalab/reproduce.py --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --repair

    # To run experiments that tune hyperparameters for ID vs OOD val experiments
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x336bc32535484f3bbad55c88bf1b05d0 --datasets amazon camelyon17 iwildcam
    python reproducibility/codalab/reproduce.py --split id_val_eval --post-tune --worksheet-uuid 0x036017edb3c74b0692831fadfe8cbf1b --datasets iwildcam 
    
    # To output results for a specific run early stopped using OOD validation results
    python reproducibility/codalab/reproduce.py --split val_eval --uuid 0xd9ceb4
    python reproducibility/codalab/reproduce.py --split test_eval --uuid 0x4516cc
     
    # To output full results of a experiment across multiple replicates
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0xa0b262fc173f43c297409a069a021496 --experiment globalwheat_erm_indist_seed
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x710beabc2aa84778a7fe21db55e24492 --experiment poverty_erm_ID_fold --id-val   
    
    # DANN on domainnent
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x13ef64a3a90842d981b6b1f566b1cc78 --experiment domainnet_real-sketch

    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x13ef64a3a90842d981b6b1f566b1cc78 --experiment fmow_dann1
    
    # To output how long it takes to download a dataset and train/eval for a given run
    python reproducibility/codalab/reproduce.py --time --uuid 0x7cc5b4
"""


class CodaLabReproducibility:
    _ADDITIONAL_DATASETS = ["bdd100k", "celebA", "waterbirds", "yelp"]
    _CIVIL_COMMENTS_ADDITIONAL_ALGORITHMS = [
        "erm_groupby-y",
        "erm_groupby-black-y",
        "groupDRO_groupby-y",
        "groupDRO_groupby-black-y",
    ]

    def __init__(self, wilds_version, all_datasets=False):
        self._wilds_version = wilds_version
        self._all_datasets = all_datasets

        # Run experiments on the main instance - https://worksheets.codalab.org
        self._run(["cl", "work", "https://worksheets.codalab.org::"])

    # TODO: remove this later if not needed -Tony
    def analyze(self, in_distribution=False):
        worksheet_uuid = self._set_worksheet()
        datasets = (
            dataset_defaults.keys()
            if not in_distribution
            else list(CORAL_HYPERPARAMETER_SEARCH_SPACE["datasets"].keys())
        )

        for dataset in datasets:
            if (
                not self._all_datasets
                and dataset in CodaLabReproducibility._ADDITIONAL_DATASETS
            ):
                continue

            dataset_results = dict()
            algorithms = (
                [key for key in algorithm_defaults.keys()]
                + CodaLabReproducibility._CIVIL_COMMENTS_ADDITIONAL_ALGORITHMS
                if dataset == "civilcomments"
                else algorithm_defaults.keys()
            )
            for algorithm in algorithms:
                dataset_results[algorithm] = []

                # Skip the additional datasets if we don't want to reproduce results for all datasets
                if (
                    (algorithm in ["IRM", "deepCORAL"] and dataset == "civilcomments")
                    or algorithm == "groupDRO"
                    or (in_distribution and algorithm != "ERM")
                ):
                    continue

                if dataset == "poverty":
                    folds = ["A", "B", "C", "D", "E"]
                    for fold in folds:
                        bundle_name = (
                            f"{dataset}_{algorithm}_fold{fold}"
                            if not in_distribution
                            else f"hp_id_{algorithm}_{dataset}_fold{fold}"
                        )
                        uuid = self._get_bundle_uuid(bundle_name, worksheet_uuid)
                        results_dfs = load_results(
                            f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                            splits=["val", "test"],
                            include_in_distribution=True,
                        )
                        dataset_results[algorithm].append(results_dfs)
                else:
                    # 10 different seeds were used for camelyon17. 3 different seeds for other datasets.
                    seeds = range(0, 10) if dataset == "camelyon17" else [0, 1, 2]
                    for seed in seeds:
                        bundle_name = (
                            f"{dataset}_{algorithm}_seed{seed}"
                            if not in_distribution
                            else f"hp_id_{algorithm}_{dataset}_seed{seed}"
                        )
                        uuid = self._get_bundle_uuid(bundle_name, worksheet_uuid)
                        results_dfs = load_results(
                            f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                            splits=["val", "test"],
                            include_in_distribution=True,
                        )
                        dataset_results[algorithm].append(results_dfs)
            compiled_results = compile_results(
                dataset, dataset_results, in_distribution
            )

            subdirectory = "main" if not in_distribution else "id_val"
            with open(
                f"reproducibility/codalab/output/{subdirectory}/{dataset}_result.json",
                "w",
            ) as f:
                json.dump(compiled_results, f, indent=4)

    def tune_hyperparameters_grid(
        self, worksheet_uuid, datasets, algorithm="deepCORAL", coarse=False
    ):
        def get_grid(params):
            # Returns the Cartesian product of the parameters to form the grid.
            grid = []
            for p in product(*params):
                grid.append(p)
            return grid

        self._set_worksheet(worksheet_uuid)
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid, datasets)
        wilds_src_uuid = self._get_bundle_uuid("wilds-unlabeled", worksheet_uuid)

        self._add_header(
            f"Hyperparameter tuning: algorithm={algorithm}, coarse={coarse}"
        )
        for dataset, dataset_uuid in datasets_uuids.items():
            dataset_fullname = self._get_field_value(dataset_uuid, "name")
            search_space = self._get_hyperparameter_search_space(algorithm)

            hyperparameters = search_space[dataset].keys()
            for hyperparameter_values in get_grid(search_space[dataset].values()):
                hyperparameter_config = dict()
                for i, hyperparameter in enumerate(hyperparameters):
                    hyperparameter_config[hyperparameter] = hyperparameter_values[i]

                self._run_experiment(
                    name=f"{dataset}_{algorithm.lower()}{'_coarse' if coarse else ''}_tune",
                    description=f"{str(hyperparameter_config)}",
                    dependencies={
                        "wilds": wilds_src_uuid,
                        dataset_fullname: dataset_uuid,
                    },
                    command=self._construct_command(
                        dataset,
                        algorithm=algorithm,
                        seed=0,
                        hyperparameters=hyperparameter_config,
                        coarse=coarse,
                    ),
                )

    def tune_hyperparameters_random(
        self,
        worksheet_uuid,
        datasets,
        algorithm="DANN",
        coarse=False,
        num_of_samples=20,
    ):
        self._set_worksheet(worksheet_uuid)
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid, datasets)
        wilds_src_uuid = self._get_bundle_uuid("wilds-unlabeled", worksheet_uuid)

        self._add_header(
            f"Hyperparameter tuning: algorithm={algorithm}, coarse={coarse}"
        )
        for dataset, dataset_uuid in datasets_uuids.items():
            dataset_fullname = self._get_field_value(dataset_uuid, "name")
            search_space = self._get_hyperparameter_search_space(algorithm)

            for _ in range(num_of_samples):
                hyperparameter_config = dict()
                for hyperparameter, values in search_space[dataset].items():
                    if len(values) == 1:
                        hyperparameter_config[hyperparameter] = values[0]
                    else:
                        hyperparameter_config[hyperparameter] = math.pow(
                            10, np.random.uniform(low=values[0], high=values[-1])
                        )

                self._run_experiment(
                    name=f"{dataset}_{algorithm.lower()}{'_coarse' if coarse else ''}_tune",
                    description=f"{str(hyperparameter_config)}",
                    dependencies={
                        "wilds": wilds_src_uuid,
                        dataset_fullname: dataset_uuid,
                    },
                    command=self._construct_command(
                        dataset,
                        algorithm=algorithm,
                        seed=0,
                        hyperparameters=hyperparameter_config,
                        coarse=coarse,
                    ),
                )

    def _run_experiment(self, name, description, dependencies, command, dry_run=False):
        commands = [
            "cl",
            "run",
            f"--name={name}",
            f"--description={description}",
            "--request-docker-image=pangwei/wilds_src:1.0",
            "--request-network",
            "--request-cpus=4",
            "--request-gpus=1",
            "--request-disk=10g",
            "--request-memory=19g",
            "--request-priority=20",
        ]
        for key, uuid in dependencies.items():
            commands.append(f"{key}:{uuid}")
        commands.append(command)
        self._run(commands, dry_run=dry_run)

    def _get_hyperparameter_search_space(self, algorithm):
        if algorithm == "deepCORAL":
            search_space = CORAL_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "DANN":
            search_space = DANN_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        return search_space

    def output_hyperparameter_tuning_results(
        self, worksheet_uuid, datasets, split, experiment_name
    ):
        self._set_worksheet(worksheet_uuid)

        # For each dataset, output the run bundle info with the best metric value for a given split
        for dataset in datasets:
            metrics = get_metrics(dataset)
            metric = metrics[0]
            print(f"Using {metric} for {split} to find the best hyperparameters...\n")
            grid_uuids = self._run(
                [
                    "cl",
                    "search",
                    experiment_name,
                    "state=ready",
                    f"host_worksheet={worksheet_uuid}",
                    ".limit=100",
                    "--uuid-only",
                ]
            ).split("\n")
            best_uuid = None
            best_metric_value = 0
            for uuid in grid_uuids:
                # CodaLab returns an empty string if there are no results
                if not uuid:
                    continue

                results_dfs = load_results(
                    f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                    splits=["val", "test"],
                    include_in_distribution=True,
                )
                val_result_df = get_early_stopped_row(
                    results_dfs[split], results_dfs[split], metrics
                )
                test_result_df = get_early_stopped_row(
                    results_dfs["test_eval"], results_dfs["val_eval"], metric
                )
                print(
                    f"uuid={uuid}, validation={val_result_df[metric]}, test={test_result_df[metric]}"
                )
                if val_result_df[metric] > best_metric_value:
                    print(
                        f"Found a new best {val_result_df[metric]} > {best_metric_value}. UUID: {uuid}"
                    )
                    best_uuid = uuid
                    best_metric_value = val_result_df[metric]

            if best_uuid:
                bundle_name = self._get_field_value(best_uuid, "name")
                bundle_description = self._get_field_value(best_uuid, "description")

                print("\n")
                print("-" * 100)
                print(
                    f"Best for {dataset} \nuuid: {best_uuid} \nName: {bundle_name} \nDescription: {bundle_description}"
                )
                print("-" * 100)
                print("\n")
            else:
                print(f"\nDid not find ready bundles for {dataset}.")

    def repair(self, worksheet_uuid):
        def fetch_bundles_by_state(state):
            return self._run(
                [
                    "cl",
                    "search",
                    "host_worksheet=%s" % worksheet_uuid,
                    f"state={state}",
                    ".limit=100",
                    "--uuid-only",
                ]
            ).split("\n")

        print("Repairing {}...".format(worksheet_uuid))
        self._run(["cl", "work", worksheet_uuid])

        for state in ["failed", "worker_offline", "killed"]:
            for uuid in fetch_bundles_by_state(state):
                if uuid:
                    self._run(["cl", "mimic", uuid])
                    self._run(["cl", "rm", uuid])

    def output_full_results(
        self, worksheet_uuid, experiment, in_distribution_val=False
    ):
        """
        Output the full results for an experiment. The output of this function is
        used as the results of the paper.

        Parameters:
            worksheet_uuid(str): Partial or full UUID of the worksheet where the
                                 results are hosted
            experiment(str): Name of the experiment (e.g. amazonv2.0_irm)
            in_distribution_val(bool): If true, use ID validation set for early stopping.
        """
        dataset = self._get_dataset_name(experiment_name=experiment)
        results = {"Result": []}

        # Search for all the replicates of an experiment
        experiment_uuids = self._run(
            [
                "cl",
                "search",
                experiment,
                "state=ready",
                "host_worksheet=%s" % worksheet_uuid,
                ".limit=10",
                "--uuid-only",
            ]
        ).split("\n")

        for uuid in experiment_uuids:
            # CodaLab returns an empty string if there are no results
            if not uuid:
                continue

            results_dfs = load_results(
                f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                splits=["val", "test"],
                include_in_distribution=True,
            )
            results["Result"].append(results_dfs)

        print(f"in_distribution_val={in_distribution_val}")
        print(
            json.dumps(
                compile_results(
                    dataset, results, in_distribution_val=in_distribution_val
                ),
                indent=4,
            )
        )
        print("\n")

    def get_result(self, uuid, split="val_eval"):
        """
        Output the results of a single run bundle.

        Parameters:
            uuid (str): Partial or full UUID of the experiment bundle.
            split (str): The split we want to show results for.
        """
        bundle_name = self._run(
            ["cl", "info", uuid, "--field=name"], print_output=False
        )
        full_uuid = self._run(["cl", "info", uuid, "--field=uuid"], print_output=False)
        results_dfs = load_results(
            f"https://worksheets.codalab.org/rest/bundles/{full_uuid}/contents/blob",
            splits=["train", "val", "test"],
            include_in_distribution=True,
        )
        sort_metrics = get_metrics(bundle_name)
        print(f"Getting early stopped row by using metrics: {sort_metrics}")
        test_result_df = get_early_stopped_row(
            results_dfs[split], results_dfs["val_eval"], sort_metrics
        )
        print(f"\n{bundle_name}:\n{test_result_df}")

    def _construct_command(
        self, dataset_name, algorithm, seed, hyperparameters, coarse=False
    ):
        command = (
            "python wilds/examples/run_expt.py --root_dir $HOME --log_dir $HOME "
            f"--dataset {dataset_name} --algorithm {algorithm} --seed {seed}"
        )
        command += " --unlabeled_split test_unlabeled"
        if coarse:
            command += " --groupby_fields from_source_domain"
        for hyperparameter, value in hyperparameters.items():
            command += f" --{hyperparameter} {value}"
        return command

    def _get_dataset_name(self, experiment_name):
        for dataset in list(dataset_defaults.keys()) + [
            "domainnet",
            "rxrx1",
            "globalwheat",
        ]:
            if dataset in experiment_name:
                return dataset
        raise ValueError(
            f"Could not find the corresponding dataset for experiment {experiment_name}"
        )

    def _get_datasets_uuids(self, worksheet_uuid, datasets):
        return {
            dataset: self._get_bundle_uuid(f"{dataset}", worksheet_uuid)
            for dataset in datasets
        }

    def _get_bundle_uuid(self, name, worksheet_uuid):
        results = self._run(
            [
                "cl",
                "search",
                name,
                "host_worksheet=%s" % worksheet_uuid,
                ".limit=1",
                "--uuid-only",
            ]
        ).split("\n")
        if len(results) != 1 or not results[0]:
            raise RuntimeError(f"Could not fetch bundle UUID for {name}")
        return results[0]

    def _set_worksheet(self, worksheet_uuid):
        self._run(["cl", "work", worksheet_uuid])

    def _get_worksheet_uuid(self, worksheet_name):
        results = self._run(
            [
                "cl",
                "wsearch",
                f"name={worksheet_name}",
                ".limit=1",
                "--uuid-only",
            ]
        ).split("\n")
        if len(results) != 1 or not results[0]:
            raise RuntimeError(f"Could not fetch UUID for worksheet: {worksheet_name}")
        return results[0]

    def _add_header(self, title, level=3):
        self._add_text("")
        self._add_text("{} {}".format("#" * level, title))
        self._add_text("")

    def _add_text(self, text):
        self._run(["cl", "add", "text", text])

    def _get_field_value(self, uuid, field):
        """
        Return the field value for a given bundle.

        Parameters:
            uuid (str): Partial or full UUID of the experiment bundle.
            field (str): Field to retrieve the value of.
        """
        return self._run(["cl", "info", uuid, f"--field={field}"], print_output=False)

    def _run(self, command, print_output=True, clean=True, dry_run=False):
        def clean_output(output):
            # Clean output to get UUID, states, etc.
            return output.strip("\n").strip()

        # If dry_run is set to True, just print the command and return back an empty output
        if dry_run:
            print("[DRY] %s" % " ".join(command))
            return ""

        print(" ".join(command))
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if print_output:
            print(process.stdout)
        return clean_output(process.stdout) if clean else process.stdout

    def time(self, uuid):
        """
        Output the the time it takes to download a dataset and train + eval.

        Parameters:
            uuid (str): Partial or full UUID of the experiment bundle.
        """
        # Since partial UUID's are allow, first retrieve the full uuid
        # and name of the bundle.
        uuid = self._get_field_value(uuid, "uuid")
        name = self._get_field_value(uuid, "name")

        # started = When the worker starts to download the datasets
        started_datetime = int(self._get_field_value(uuid, "started"))
        # last_updated = When the worker completes the job
        last_updated_time = int(self._get_field_value(uuid, "last_updated"))
        # Calculate the total time for the experiment to complete (in seconds)
        total_time_seconds = last_updated_time - started_datetime

        # time = Total time (in seconds) it takes to train and evaluate
        runtime = float(self._get_field_value(uuid, "time"))
        runtime_hours = int(runtime // 3600)
        remaining_seconds = runtime % 3600
        runtime_minutes = int(remaining_seconds / 60)
        # Calculate the total time it takes to download the dataset
        download_time_minutes = round((total_time_seconds - runtime) / 60)

        # Output times
        print(f"\nRun name: {name}")
        print(f"UUID: {uuid}")
        print(f"\nDataset: {self._get_dataset_name(name)}")
        print(f"Download time: {download_time_minutes} minutes")
        print(f"Train + eval time: {runtime_hours}h{runtime_minutes}m\n")


def main():
    print(args)

    reproducibility = CodaLabReproducibility(args.version, args.all_datasets)
    if args.tune_hyperparameters:
        if args.random_search:
            reproducibility.tune_hyperparameters_random(
                args.worksheet_uuid, args.datasets, args.algorithm, args.coarse
            )
        else:
            reproducibility.tune_hyperparameters_grid(
                args.worksheet_uuid, args.datasets, args.algorithm, args.coarse
            )
    elif args.post_tune:
        reproducibility.output_hyperparameter_tuning_results(
            args.worksheet_uuid, args.datasets, args.split, args.experiment
        )
    elif args.repair:
        reproducibility.repair(worksheet_uuid=args.worksheet_uuid)
    elif args.output_results:
        reproducibility.output_full_results(
            args.worksheet_uuid, args.experiment, args.id_val
        )
    elif args.time:
        reproducibility.time(args.uuid)
    elif args.uuid:
        reproducibility.get_result(args.uuid, args.split)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce results of WILDS")
    parser.add_argument(
        "--version",
        type=str,
        help="Version of WILDS to reproduce results for",
        default="v1.0",
    )
    parser.add_argument(
        "--worksheet-name",
        type=str,
        help="Name of the CodaLab worksheet to reproduce the results on.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Whether to run experiments on all the datasets (default to false).",
    )
    parser.add_argument(
        "--datasets", nargs="+", help="List of datasets to run experiments on."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        help="Name of the algorithm",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Whether to run with coarse-grained domains instead of fine-grained domains (defaults to false).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Whether to run repair worksheets with hyperparameter tuning experiments (defaults to false).",
    )
    parser.add_argument(
        "--id-val",
        action="store_true",
        help="ID val results",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Whether to run the ID validation hyperparameter tuning experiments (default to false).",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        help="Whether to run random search to tune hyperparameters (default to false).",
    )
    parser.add_argument(
        "--post-tune",
        action="store_true",
        help="Follow-up to ID validation hyperparameter tuning experiments (default to false).",
    )
    parser.add_argument(
        "--output-results",
        action="store_true",
        help="Output results for any set of WILDS experiments (default to false).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to output full results for (e.g. civilcomments_distilbert_erm).",
    )
    parser.add_argument(
        "--worksheet-uuid",
        type=str,
        help="UUID of the worksheet where the experiment runs are hosted",
    )
    parser.add_argument(
        "--uuid",
        type=str,
        help="When specified, output results for this specific run. A partial UUID is acceptable.",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="The split we want to show results for (e.g. val_eval).",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Whether to time a download + train/eval time of a run (defaults to false).",
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
