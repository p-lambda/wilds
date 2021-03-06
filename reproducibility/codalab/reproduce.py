import argparse
import json
import subprocess
from itertools import product

from examples.configs.algorithm import algorithm_defaults
from examples.configs.datasets import dataset_defaults
from reproducibility.codalab.hyperparameter_search_space import (
    HYPERPARAMETER_SEARCH_SPACE,
)
from reproducibility.codalab.util.analysis_utils import (
    compile_results,
    get_early_stopped_row,
    get_metrics,
    load_results,
)

"""
Reproduce results of the WILDS paper in CodaLab.

Usage:
    python reproducibility/codalab/reproduce.py <args>
    
Example Usage:
    # To run experiments that tune hyperparameters for ID vs OOD val experiments
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x336bc32535484f3bbad55c88bf1b05d0 --datasets amazon camelyon17 iwildcam
    python reproducibility/codalab/reproduce.py --split id_val_eval --post-tune --worksheet-uuid 0x336bc32535484f3bbad55c88bf1b05d0 --datasets fmow  
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x336bc32535484f3bbad55c88bf1b05d0 --experiment fmow_erm_seed
    
    # To output results for OOD hyperparameter tuning
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x036017edb3c74b0692831fadfe8cbf1b --datasets iwildcam --experiment iwildcam2.0_erm_tune
    
    # To output results for a specific run early stopped using OOD validation results
    python reproducibility/codalab/reproduce.py --split val_eval --uuid 0x854a4f
    
    # To output full results of a experiment across multiple replicates 
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x15602bb702b64071ae98a4d5d8808068 --experiment amazonv2.0_erm_seed
    
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

    def reproduce_main_results(self, worksheet_uuid):
        worksheet_uuid = self._set_worksheet(worksheet_uuid)
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid)
        wilds_src_uuid = self._get_bundle_uuid("wilds", worksheet_uuid)

        for algorithm in algorithm_defaults.keys():
            self._add_header(algorithm)

            for dataset in dataset_defaults.keys():
                # Skip the additional datasets if we don't want to reproduce results for all datasets
                if (
                    not self._all_datasets
                    and dataset in CodaLabReproducibility._ADDITIONAL_DATASETS
                ) or (algorithm in ["IRM", "deepCORAL"] and dataset == "civilcomments"):
                    continue

                self._add_header(dataset, level=5)
                dataset_fullname = f"{dataset}_{self._wilds_version}"
                dataset_uuid = datasets_uuids[dataset]
                if dataset == "poverty":
                    folds = ["A", "B", "C", "D", "E"]
                    for fold in folds:
                        self._run(
                            [
                                "cl",
                                "run",
                                f"--name=${dataset}_${algorithm}_fold${fold}",
                                "--request-docker-image=codalabrunner/wilds_test:0.1",  # TODO: hardcoded
                                f"--request-cpus=4",
                                "--request-gpus=1",
                                "--request-disk=10g",
                                "--request-memory=54g",
                                "--request-network",
                                f"wilds:{wilds_src_uuid}",
                                f"{dataset_fullname}:{dataset_uuid}",
                                self._construct_command(
                                    dataset,
                                    algorithm=algorithm,
                                    seed=0,
                                    hyperparameters={
                                        "dataset_kwargs": f"oracle_training_set=True fold=${fold}"
                                    },
                                ),
                            ]
                        )
                else:
                    # Use 10 different seeds for camelyon17. Use just 3 different seeds for other datasets.
                    seeds = range(0, 10) if dataset == "camelyon17" else [0, 1, 2]
                    for seed in seeds:
                        self._run(
                            [
                                "cl",
                                "run",
                                f"--name=${dataset}_${algorithm}_seed${seed}",
                                "--request-docker-image=codalabrunner/wilds_test:0.1",  # TODO: hardcoded
                                f"--request-cpus=8",
                                "--request-gpus=1",
                                "--request-disk=10g",
                                "--request-memory=54g",
                                "--request-network",
                                f"wilds:{wilds_src_uuid}",
                                f"{dataset_fullname}:{dataset_uuid}",
                                self._construct_command(
                                    dataset,
                                    algorithm=algorithm,
                                    seed=seed,
                                    hyperparameters={},
                                ),
                            ]
                        )

    def analyze(self, in_distribution=False):
        worksheet_uuid = self._set_worksheet()
        datasets = (
            dataset_defaults.keys()
            if not in_distribution
            else list(HYPERPARAMETER_SEARCH_SPACE["datasets"].keys())
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

    def tune_hyperparameters_in_distribution(self, worksheet_uuid, datasets):
        def get_grid(params):
            # Returns the Cartesian product of the parameters to form the grid.
            grid = []
            for p in product(*params):
                grid.append(p)
            return grid

        self._set_worksheet(worksheet_uuid)
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid, datasets)
        wilds_src_uuid = self._get_bundle_uuid("wilds", worksheet_uuid)

        for dataset, dataset_uuid in datasets_uuids.items():
            dataset_fullname = self._get_field_value(dataset_uuid, "name")
            hyperparameters = HYPERPARAMETER_SEARCH_SPACE["datasets"][dataset].keys()

            for hyperparameter_values in get_grid(
                HYPERPARAMETER_SEARCH_SPACE["datasets"][dataset].values()
            ):
                hyperparameter_config = dict()
                for i, hyperparameter in enumerate(hyperparameters):
                    hyperparameter_config[hyperparameter] = hyperparameter_values[i]
                self._run(
                    [
                        "cl",
                        "run",
                        f"--name=hyperparameters_id_erm_{dataset_fullname}",
                        f"--description={str(hyperparameter_config)}",
                        "--request-docker-image=codalabrunner/wilds_test:0.1",  # TODO: hardcoded
                        "--request-network",
                        f"--request-cpus=8",
                        "--request-gpus=1",
                        "--request-disk=10g",
                        "--request-memory=54g",
                        f"wilds:{wilds_src_uuid}",
                        f"{dataset_fullname}:{dataset_uuid}",
                        self._construct_command(
                            dataset,
                            algorithm="ERM",
                            seed=0,
                            hyperparameters=hyperparameter_config,
                        ),
                    ],
                )

    def output_hyperparameters_tuning_results(
        self, worksheet_uuid, datasets, split, experiment_name=None
    ):
        self._set_worksheet(worksheet_uuid)

        # For each dataset, output the run bundle info with the best metric value for a given split
        for dataset in datasets:
            metrics = get_metrics(dataset)
            metric = metrics[0]
            print(f"Using {metric} for {split} to find the best hyperparameters...\n")

            # If experiment_name is not specified just default to ID naming scheme
            if not experiment_name:
                experiment_name = f"hyperparameters_id_erm_{dataset}"

            grid_uuids = self._run(
                [
                    "cl",
                    "search",
                    experiment_name,
                    "state=ready",
                    f"host_worksheet={worksheet_uuid}",
                    ".limit=50",
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
                    splits=["val"],
                    include_in_distribution=True,
                )
                val_result_df = get_early_stopped_row(
                    results_dfs[split], results_dfs[split], metrics
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

    def output_full_results(self, worksheet_uuid, experiment):
        """
        Output the full results for an experiment. The output of this function is
        used as the results of the paper.

        Parameters:
            worksheet_uuid(str): Partial or full UUID of the worksheet where the
                                 results are hosted
            experiment(str): Name of the experiment (e.g. amazonv2.0_irm)
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

        print(json.dumps(compile_results(dataset, results), indent=4))
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
            splits=["val", "test"],
            include_in_distribution=True,
        )
        sort_metrics = get_metrics(bundle_name)
        print(f"Getting early stopped row by using metrics: {sort_metrics}")
        test_result_df = get_early_stopped_row(
            results_dfs[split], results_dfs["val_eval"], sort_metrics
        )
        print(f"\n{bundle_name}:\n{test_result_df}")

    def _construct_command(self, dataset_name, algorithm, seed, hyperparameters):
        command = (
            "python wilds/examples/run_expt.py --root_dir $HOME --log_dir $HOME "
            f"--dataset {dataset_name} --algorithm {algorithm} --seed {seed}"
        )
        for hyperparameter, value in hyperparameters.items():
            command += f" --{hyperparameter} {value}"
        return command

    def _get_dataset_name(self, experiment_name):
        for dataset in dataset_defaults.keys():
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
    reproducibility = CodaLabReproducibility(args.version, args.all_datasets)
    if args.analyze:
        reproducibility.analyze(args.id_val)
    elif args.tune_hyperparameters:
        reproducibility.tune_hyperparameters_in_distribution(
            args.worksheet_uuid, args.datasets
        )
    elif args.post_tune:
        reproducibility.output_hyperparameters_tuning_results(
            args.worksheet_uuid, args.datasets, args.split, args.experiment
        )
    elif args.output_results:
        reproducibility.output_full_results(args.worksheet_uuid, args.experiment)
    elif args.time:
        reproducibility.time(args.uuid)
    elif args.uuid:
        reproducibility.get_result(args.uuid, args.split)
    else:
        # If the other flags are not set, just reproduce the main results of WILDS.
        reproducibility.reproduce_main_results(args.worksheet_uuid)
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
        "--analyze",
        action="store_true",
        help="Analyze results",
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
