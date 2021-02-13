import argparse
import json
import subprocess
from itertools import product

from examples.configs.algorithm import algorithm_defaults
from examples.configs.datasets import dataset_defaults
from reproducibility.codalab.hyperparameter_search_space import (
    ID_HYPERPARAMETER_SEARCH_SPACE,
)
from reproducibility.codalab.util.analysis_utils import (
    compile_results,
    get_early_stopped_row,
    get_metrics,
    load_results,
)

"""
Reproduce results of the WILDS paper in CodaLab.

[TODO's]
- Generate results of paper
- ID val ERM

Example usage:

    python reproducibility/codalab/reproduce_results.py 
"""


class CodaLabReproducibility:
    _GROUP_NAME = "wilds-admins"
    _ADDITIONAL_DATASETS = ["bdd100k", "celebA", "waterbirds", "yelp"]
    _CIVIL_COMMENTS_ADDITIONAL_ALGORITHMS = [
        "erm_groupby-y",
        "erm_groupby-black-y",
        "groupDRO_groupby-y",
        "groupDRO_groupby-black-y",
    ]

    def __init__(self, wilds_version, worksheet_name, all_datasets=False):
        self._wilds_version = wilds_version
        self._worksheet_name = (
            worksheet_name if worksheet_name else f"wilds-results-{wilds_version}"
        )
        self._all_datasets = all_datasets

        # Run experiments on the main instance - https://worksheets.codalab.org
        self._run(["cl", "work", "https://worksheets.codalab.org::"])

    def reproduce_main_results(self):
        worksheet_uuid = self._set_worksheet()
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
                                f"--exclude-patterns={dataset_fullname}",
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
                                f"--request-cpus=4",
                                "--request-gpus=1",
                                "--request-disk=10g",
                                "--request-memory=54g",
                                "--request-network",
                                f"--exclude-patterns={dataset_fullname}",
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
            # TODO: ADD one offs -Tony: GroupDro and ID

    def analyze(self, in_distribution=False):
        worksheet_uuid = self._set_worksheet()
        datasets = (
            dataset_defaults.keys()
            if not in_distribution
            else list(ID_HYPERPARAMETER_SEARCH_SPACE["datasets"].keys())
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

    def tune_hyperparameters_id(self):
        def get_grid(params):
            grid = []
            for p in product(*params):
                grid.append(p)
            return grid

        worksheet_uuid = self._set_worksheet()
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid)
        wilds_src_uuid = self._get_bundle_uuid("wilds", worksheet_uuid)

        for dataset, dataset_uuid in datasets_uuids.items():
            # The following datasets have in-distribution val sets and has been hyperparameter tuned
            if dataset not in ID_HYPERPARAMETER_SEARCH_SPACE["datasets"]:
                continue

            hyperparameters = ID_HYPERPARAMETER_SEARCH_SPACE["datasets"][dataset].keys()
            dataset_fullname = f"{dataset}_{self._wilds_version}"
            for hyperparameter_values in get_grid(
                ID_HYPERPARAMETER_SEARCH_SPACE["datasets"][dataset].values()
            ):
                hyperparameter_result = dict()
                for i, hyperparameter in enumerate(hyperparameters):
                    hyperparameter_result[hyperparameter] = hyperparameter_values[i]
                print(hyperparameter_result)
                self._run(
                    [
                        "cl",
                        "run",
                        f"--name=hyperparameters_id_erm_{dataset}",
                        f"--description={str(hyperparameter_result)}",
                        "--request-docker-image=codalabrunner/wilds_test:0.1",  # TODO: hardcoded
                        f"--request-cpus={8 if dataset == 'amazon' else 4}",
                        "--request-gpus=1",
                        "--request-disk=10g",
                        "--request-memory=54g",
                        "--request-network",
                        f"--exclude-patterns={dataset_fullname}",
                        f"wilds:{wilds_src_uuid}",
                        f"{dataset_fullname}:{dataset_uuid}",
                        self._construct_command(
                            dataset,
                            algorithm="ERM",
                            seed=0,
                            hyperparameters=hyperparameter_result,
                        ),
                    ]
                )

    def post_tune_hyperparameters_id(self):
        worksheet_uuid = self._set_worksheet()

        # The following datasets have in-distribution val sets and has been hyperparameter tuned
        for dataset in ID_HYPERPARAMETER_SEARCH_SPACE["datasets"].keys():
            metric = get_metrics(dataset)[0]
            grid_uuids = self._run(
                [
                    "cl",
                    "search",
                    f"name=hyperparameters_id_erm_{dataset}",
                    "state=ready",
                    "host_worksheet=%s" % worksheet_uuid,
                    ".limit=100",
                    ".shared",
                    "--uuid-only",
                ]
            ).split("\n")
            best_uuid = None
            best_accuracy = 0
            for uuid in grid_uuids:
                results_dfs = load_results(
                    f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                    splits=["val"],
                    include_in_distribution=True,
                )
                id_val_result_df = get_early_stopped_row(
                    results_dfs["id_val_eval"], results_dfs["id_val_eval"], [metric]
                )
                if id_val_result_df[metric] > best_accuracy:
                    best_uuid = uuid
                    best_accuracy = id_val_result_df[metric]

            print(f"Best for {dataset} uuid: {best_uuid}")
        print("\nTODO: run the best manually for now")

    def output_results(self):
        # TODO: hardcoded these for now for speeding up BERT -Tony
        worksheet_uuid = "0xb9e5615b78924cb48273f80b746c9fe7"
        for dataset in ["amazon", "civilcomments"]:
            print(f"-- {dataset} --")
            metrics = get_metrics(dataset)
            sort_metric = metrics[0]
            experiment_uuids = self._run(
                [
                    "cl",
                    "search",
                    f"{dataset}_erm_frac",
                    "state=ready",
                    "host_worksheet=%s" % worksheet_uuid,
                    ".limit=100",
                    "--uuid-only",
                ]
            ).split("\n")

            for uuid in experiment_uuids:
                bundle_name = self._run(
                    ["cl", "info", uuid, "--field=name"], print_output=False
                )
                results_dfs = load_results(
                    f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob",
                    splits=["val", "test"],
                    include_in_distribution=True,
                )
                test_result_df = get_early_stopped_row(
                    results_dfs["test_eval"], results_dfs["val_eval"], [sort_metric]
                )

                log_output = f"{bundle_name.split('_')[-1]}"
                for metric in metrics:
                    log_output += f" {metric}={test_result_df[metric]}"
                print(log_output)
            print("\n")

    def _construct_command(self, dataset_name, algorithm, seed, hyperparameters):
        command = (
            "python wilds/examples/run_expt.py --root_dir $HOME --log_dir $HOME "
            f"--dataset {dataset_name} --algorithm {algorithm} --seed {seed}"
        )
        for hyperparameter, value in hyperparameters.items():
            command += f" --{hyperparameter} {value}"
        return command

    def _get_datasets_uuids(self, worksheet_uuid):
        return {
            dataset: self._get_bundle_uuid(
                f"{dataset}_{self._wilds_version}", worksheet_uuid
            )
            for dataset in dataset_defaults.keys()
            if dataset not in CodaLabReproducibility._ADDITIONAL_DATASETS
            and dataset != "ogb-molpcba"
        }

    def _get_bundle_uuid(self, name, worksheet_uuid):
        results = self._run(
            [
                "cl",
                "search",
                f"name={name}",
                "host_worksheet=%s" % worksheet_uuid,
                ".limit=1",
                ".shared",
                "--uuid-only",
            ]
        ).split("\n")
        if len(results) != 1 or not results[0]:
            raise RuntimeError(f"Could not fetch bundle UUID for {name}")
        return results[0]

    def _set_worksheet(self):
        worksheet_uuid = self._get_worksheet_uuid()
        self._run(["cl", "work", worksheet_uuid])
        return worksheet_uuid

    def _get_worksheet_uuid(self, worksheet_name):
        results = self._run(
            [
                "cl",
                "wsearch",
                f"name={self._worksheet_name}",
                ".limit=1",
                ".shared",
                "--uuid-only",
            ]
        ).split("\n")
        if len(results) != 1 or not results[0]:
            raise RuntimeError(
                f"Could not fetch UUID for worksheet: {self._worksheet_name}"
            )
        return results[0]

    def _give_access(self, entity, for_worksheet=False):
        perm_command = "wperm" if for_worksheet else "perm"
        self._run(
            ["cl", perm_command, entity, CodaLabReproducibility._GROUP_NAME, "all"]
        )

    def _add_header(self, title, level=3):
        self._add_text("")
        self._add_text("{} {}".format("#" * level, title))
        self._add_text("")

    def _add_text(self, text):
        self._run(["cl", "add", "text", text])

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


def main():
    reproducibility = CodaLabReproducibility(
        args.version, args.worksheet_name, args.all_datasets
    )
    if args.analyze:
        reproducibility.analyze(args.id_val)
    elif args.tune_hyperparameters:
        reproducibility.tune_hyperparameters_id()
    elif args.post_tune:
        reproducibility.post_tune_hyperparameters_id()
    elif args.output_results:
        reproducibility.output_results()
    else:
        # If the other flags are not set, just reproduce the main results of WILDS.
        reproducibility.reproduce_main_results()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce results of WILDS")
    parser.add_argument(
        "--version",
        type=str,
        help="Version of WILDS to reproduce results for",
        default="v1.0",
    )
    parser.add_argument(
        "--worksheet_name",
        type=str,
        help="Name of the CodaLAb worksheet to reproduce the results on.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Whether to run experiments on all the datasets (default to false).",
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

    # Parse args and run this script
    args = parser.parse_args()
    main()
