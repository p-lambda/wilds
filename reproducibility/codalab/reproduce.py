import argparse
import json
import math
import pdb
import subprocess
from itertools import product

import numpy as np

from examples.configs.datasets import dataset_defaults
from reproducibility.codalab.hyperparameter_search_space import (
    OGB,
    MAX_BATCH_SIZES,
    ERM_HYPERPARAMETER_SEARCH_SPACE,
    ERM_AUGMENT_HYPERPARAMETER_SEARCH_SPACE,
    ERM_ORACLE_HYPERPARAMETER_SEARCH_SPACE,
    CORAL_HYPERPARAMETER_SEARCH_SPACE,
    DANN_HYPERPARAMETER_SEARCH_SPACE,
    FIXMATCH_HYPERPARAMETER_SEARCH_SPACE,
    PSEUDOLABEL_HYPERPARAMETER_SEARCH_SPACE,
    NOISY_STUDENT_HYPERPARAMETER_SEARCH_SPACE,
    NOISY_STUDENT_TEACHERS,
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
    # To tune for ERM runs
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x36ab18e5c43c480f8766a3351d3efad2 --datasets ogb-molpcba --algorithm ERM --random --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets camelyon17 --experiment fmow_erm_tune 
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets poverty --algorithm ERM --random --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets camelyon17--experiment fmow_ermaugment_tune
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets amazon --algorithm ERMOracle --random --gpus 1 --unlabeled-split test_unlabeled --dry-run

    # To tune for multi-gpu runs
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets domainnet --algorithm PseudoLabel --random --gpus 1 --unlabeled-split test_unlabeled --weak --dry-run
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets domainnet --algorithm FixMatch --random --gpus 1 --unlabeled-split test_unlabeled --weak --dry-run
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets iwildcam --algorithm FixMatch --random --gpus 1 --unlabeled-split extra_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets globalwheat --algorithm NoisyStudent --random --gpus 1 --unlabeled-split test_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm FixMatch --random --gpus 1 --unlabeled-split test_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_pseudolabel_tune 

    # To tune model hyperparameters for Unlabeled WILDS
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets domainnet --algorithm DANN --random --coarse --unlabeled-split test_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_deepcoral_tune
  
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets iwildcam --algorithm deepCORAL --random --coarse --unlabeled-split extra_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x5eebc93ea19b4dd99aa68871d18d7cb2 --datasets fmow --experiment fmow_dann_coarse_valunlabeled_tune	

    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm DANN --random --unlabeled-split test_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --experiment fmow_dann_tune
    
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm deepCORAL --random --unlabeled-split val_unlabeled
    
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --datasets fmow --algorithm DANN --random --coarse --unlabeled-split test_unlabeled --dry-run
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0x5eebc93ea19b4dd99aa68871d18d7cb2 --datasets fmow --experiment fmow_dann_coarse_trainunlabeled_tune
    
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0xdc42650973ef4c4e9db3ed356de876ee --datasets amazon --experiment amazon_dann_coarse_tune
    
    python reproducibility/codalab/reproduce.py --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --repair

    # To run experiments that tune hyperparameters for ID vs OOD val experiments
    python reproducibility/codalab/reproduce.py --tune-hyperparameters --worksheet-uuid 0x336bc32535484f3bbad55c88bf1b05d0 --datasets amazon camelyon17 iwildcam
    python reproducibility/codalab/reproduce.py --split id_val_eval --post-tune --worksheet-uuid 0x036017edb3c74b0692831fadfe8cbf1b --datasets iwildcam
    
    python reproducibility/codalab/reproduce.py --split val_eval --post-tune --worksheet-uuid 0xa0b262fc173f43c297409a069a021496 --datasets globalwheat --experiment globalwheat_erm_grid
    
    # To output results for a specific run early stopped using OOD validation results
    python reproducibility/codalab/reproduce.py --split val_eval --uuid 0xd9ceb4
    python reproducibility/codalab/reproduce.py --split test_eval --uuid 0x4516cc
     
    # To output full results of a experiment across multiple replicates
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x6eff199eaf61473291730321951dca7d --experiment fmow_dann_coarse_seed
    python3 reproducibility/codalab/reproduce.py --output --local --path /u/scr/nlp/dro/fixmatch_domainnet_logs --experiment domainnet_nlp_runs
    
    # DANN on domainnent
    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x63397d8cb2fc463c80707b149c2d90d1 --experiment domainnet_real-sketch

    python reproducibility/codalab/reproduce.py --output --worksheet-uuid 0x13ef64a3a90842d981b6b1f566b1cc78 --experiment fmow_dann1
    
    # To output how long it takes to download a dataset and train/eval for a given run
    python reproducibility/codalab/reproduce.py --time --uuid 0x7cc5b4
"""


class CodaLabReproducibility:
    _HAS_SEPARATE_UNLABELED_BUNDLE = [
        "camelyon17",
        "civilcomments",
        "iwildcam",
        "poverty",
        "globalwheat",
    ]

    def __init__(self, local=False):
        # Run experiments on the main instance - https://worksheets.codalab.org
        if not local:
            self._run(["cl", "work", "https://worksheets.codalab.org::"])

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
            for i, hyperparameter_values in enumerate(
                get_grid(search_space[dataset].values())
            ):
                hyperparameter_config = dict()
                for i, hyperparameter in enumerate(hyperparameters):
                    hyperparameter_config[hyperparameter] = hyperparameter_values[i]

                experiment_name = f"{dataset}_{algorithm.lower()}{'_coarse' if coarse else ''}_tune{i}"
                self._run_experiment(
                    name=experiment_name,
                    dataset=dataset,
                    description=f"{str(hyperparameter_config)}",
                    dependencies={
                        "wilds": wilds_src_uuid,
                        dataset_fullname: dataset_uuid,
                    },
                    command=self._construct_command(
                        dataset,
                        experiment_name,
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
        num_of_samples=10,
        unlabeled_split=None,
        dry_run=False,
        gpus=1,
        weak=False,
    ):
        self._set_worksheet(worksheet_uuid)
        datasets_uuids = self._get_datasets_uuids(worksheet_uuid, datasets)
        wilds_src_uuid = self._get_bundle_uuid("wilds-unlabeled", worksheet_uuid)
        wandb_api_key_uuid = self._get_bundle_uuid("wandb_api_key.txt", worksheet_uuid)

        self._add_header(
            f"Hyperparameter tuning: algorithm={algorithm}, coarse={coarse}",
            dry_run=dry_run,
        )
        for dataset, dataset_uuid in datasets_uuids.items():
            search_space = self._get_hyperparameter_search_space(algorithm)

            if (
                unlabeled_split
                and dataset in CodaLabReproducibility._HAS_SEPARATE_UNLABELED_BUNDLE
            ):
                unlabeled_dataset_uuid = self._get_bundle_uuid(
                    f"{dataset}_unlabeled", worksheet_uuid
                )
                unlabeled_dataset_fullname = self._get_field_value(
                    unlabeled_dataset_uuid, "name"
                )
            else:
                unlabeled_dataset_uuid = None
                unlabeled_dataset_fullname = None

            for i in range(num_of_samples):
                hyperparameter_config = dict()
                if gpus == 1 and "ERM" not in algorithm:
                    hyperparameter_config["gradient_accumulation_steps"] = 4
                if weak:
                    hyperparameter_config["additional_train_transform"] = "weak"

                for hyperparameter, values in search_space[dataset].items():
                    if hyperparameter == "n_epochs":
                        continue

                    if hyperparameter == "unlabeled_batch_size_frac":
                        max_batch_size = MAX_BATCH_SIZES[dataset] * gpus
                        index = np.random.choice(range(len(values)))
                        unlabeled_frac = values[index]
                        unlabeled_batch_size = int(unlabeled_frac * max_batch_size)
                        hyperparameter_config[
                            "unlabeled_batch_size"
                        ] = unlabeled_batch_size
                        hyperparameter_config["batch_size"] = (
                            max_batch_size - unlabeled_batch_size
                        )
                        if "n_epochs" in search_space[dataset]:
                            hyperparameter_config["n_epochs"] = search_space[dataset][
                                "n_epochs"
                            ][index]
                    else:
                        if len(values) == 1:
                            hyperparameter_config[hyperparameter] = values[0]
                        elif hyperparameter == "self_training_threshold":
                            hyperparameter_config[hyperparameter] = np.random.uniform(
                                low=values[0], high=values[-1]
                            )
                        else:
                            hyperparameter_config[hyperparameter] = math.pow(
                                10, np.random.uniform(low=values[0], high=values[-1])
                            )
                            if hyperparameter == "dann_classifier_lr":
                                hyperparameter_config["dann_featurizer_lr"] = (
                                    hyperparameter_config[hyperparameter] / 10.0
                                )

                dependencies = {
                    "wilds": wilds_src_uuid,
                    "wandb_api_key.txt": wandb_api_key_uuid,
                }
                if dataset_uuid:
                    dataset_fullname = self._get_field_value(dataset_uuid, "name")
                    dependencies[dataset_fullname] = dataset_uuid

                if algorithm == "NoisyStudent":
                    dependencies["teacher"] = NOISY_STUDENT_TEACHERS[dataset]

                if unlabeled_dataset_uuid:
                    dependencies[unlabeled_dataset_fullname] = unlabeled_dataset_uuid

                experiment_name = (
                    f"{dataset}_{algorithm.lower()}{'_coarse' if coarse else ''}{'_weak' if weak else ''}"
                )
                if unlabeled_split:
                    experiment_name += f"_{unlabeled_split.replace('_', '')}"
                experiment_name += f"_tune{i}"
                self._run_experiment(
                    name=experiment_name,
                    dataset=dataset,
                    description=f"{str(hyperparameter_config)}",
                    dependencies=dependencies,
                    command=self._construct_command(
                        dataset,
                        experiment_name,
                        algorithm="ERM" if "ERM" in algorithm else algorithm,
                        seed=0,
                        hyperparameters=hyperparameter_config,
                        coarse=coarse,
                        unlabeled_split=unlabeled_split,
                        gpus=gpus,
                    ),
                    gpus=gpus,
                    dry_run=dry_run,
                )

    def _run_experiment(
        self, name, dataset, description, dependencies, command, gpus=1, dry_run=False
    ):
        if "noisystudent" in name:
            # Training the teacher requires only a single gpu
            gpus = 1

        if gpus == 1:
            cpus = 4
            memory_gb = 96 if dataset == OGB else 16
        else:
            cpus = 8
            memory_gb = 32

        commands = [
            "cl",
            "run",
            f"--name={name}",
            f"--description={description}",
            "--request-docker-image=codalabrunner/wilds_unlabeled:1.1",
            "--request-network",
            f"--request-cpus={cpus}",
            f"--request-gpus={gpus}",
            "--request-disk=20g",
            f"--request-memory={memory_gb}g",
            "--request-priority=1",
            "--request-queue=cluster",
        ]
        if dataset == OGB:
            commands.append("--exclude-patterns=data")

        if gpus > 1:
            commands.append("--request-queue=multi")

        for key, uuid in dependencies.items():
            commands.append(f"{key}:{uuid}")
        commands.append(command)
        self._run(commands, dry_run=dry_run)

    def _get_hyperparameter_search_space(self, algorithm):
        if algorithm == "ERM":
            search_space = ERM_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "ERMAugment":
            search_space = ERM_AUGMENT_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "ERMOracle":
            search_space = ERM_ORACLE_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "deepCORAL":
            search_space = CORAL_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "DANN":
            search_space = DANN_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "FixMatch":
            search_space = FIXMATCH_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "PseudoLabel":
            search_space = PSEUDOLABEL_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        elif algorithm == "NoisyStudent":
            search_space = NOISY_STUDENT_HYPERPARAMETER_SEARCH_SPACE["datasets"]
        else:
            raise ValueError(
                f"Hyperparameter tuning for {algorithm} is not yet supported."
            )
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
                    "state=ready,killed,worker_offline",
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
                    f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob"
                    f"{'/student2' if 'noisystudent' in experiment_name else ''}",
                    splits=["val", "test"],
                    include_in_distribution=True,
                )
                val_result_df = get_early_stopped_row(
                    results_dfs[split], results_dfs[split], metrics
                )
                test_result_df = get_early_stopped_row(
                    results_dfs["test_eval"],
                    results_dfs["val_eval"],
                    metric,
                    log=True,
                )
                bundle_description = self._get_field_value(uuid, "description")
                print(
                    f"uuid={uuid}, validation={val_result_df[metric]}, test={test_result_df[metric]}, description={bundle_description}"
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
                "state=ready,killed,failed",
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
                f"https://worksheets.codalab.org/rest/bundles/{uuid}/contents/blob"
                f"{'/student2' if 'noisystudent' in experiment else ''}",
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

    def output_full_results_local(self, experiment, path, in_distribution_val=False):
        """
        Output the full results for an experiment using experiments stored locally.

        Parameters:
            experiment(str): Name of the experiment (e.g. amazonv2.0_irm)
            path(str): path to results stored locally
            in_distribution_val(bool): If true, use ID validation set for early stopping.
        """
        dataset = self._get_dataset_name(experiment_name=experiment)
        results_dfs = load_results(
            path,
            splits=["val", "test"],
            include_in_distribution=True,
        )
        results = {"Result": [results_dfs]}
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
        self,
        dataset_name,
        experiment_name,
        algorithm,
        seed,
        hyperparameters,
        coarse=False,
        unlabeled_split=None,
        gpus=1,
    ):
        if algorithm == "NoisyStudent":
            executable = (
                f"noisy_student_wrapper.py 2 teacher/{dataset_name}_"
                f"{'seed:0' if dataset_name != 'poverty' else 'fold:A'}_epoch:best_model.pth "
            )
        else:
            executable = "run_expt.py"
        command = (
            f"python -Wi wilds/examples/{executable} --root_dir $HOME{'/data' if dataset_name == OGB else ''}"
            f" --log_dir $HOME --dataset {dataset_name} --algorithm {algorithm} --seed {seed}"
        )
        if dataset_name == OGB:
            command = (
                "pip install --force-reinstall git+https://github.com/pyg-team/pytorch_geometric.git && "
                + command
            )

        if unlabeled_split:
            command += f" --unlabeled_split {unlabeled_split}"
        if coarse:
            command += " --groupby_fields from_source_domain"
        for hyperparameter, value in hyperparameters.items():
            command += f" --{hyperparameter} {value}"

        # Configure Multi-GPU
        if gpus > 1 and algorithm != "NoisyStudent":
            gpu_indices = [str(gpu) for gpu in range(gpus)]
            command += f" --device {' '.join(gpu_indices)}"

        if dataset_name not in ["amazon", "civilcomments"]:
            command += f" --loader_kwargs num_workers=4 pin_memory=True"
            if unlabeled_split != None:
                command += f" --unlabeled_loader_kwargs num_workers=8 pin_memory=True"

        # Always download ogb-molpcba dataset
        if dataset_name == OGB:
            command += f" --download"

        # Configure wandb
        # Disable pushing to WandB for Amazon - we're hitting retry loops when pushing metrics at the end of the run
        if dataset_name != "amazon":
            command += (
                f" --use_wandb --wandb_api_key_path wandb_api_key.txt --wandb_kwargs"
                f" entity=wilds project={algorithm.lower()}-{dataset_name.lower()}"
                f" group={experiment_name}_gpus{gpus}_paper"
            )
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

    def _get_datasets_uuids(self, worksheet_uuid, datasets, unlabeled=False):
        if datasets == [OGB]:
            return {datasets[0]: ""}
        return {
            dataset: self._get_bundle_uuid(
                f"{dataset}_unlabeled" if unlabeled else f"{dataset}_v", worksheet_uuid
            )
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

    def _add_header(self, title, level=3, dry_run=False):
        self._add_text("", dry_run=dry_run)
        self._add_text("{} {}".format("#" * level, title), dry_run=dry_run)
        self._add_text("", dry_run=dry_run)

    def _add_text(self, text, dry_run=False):
        self._run(["cl", "add", "text", text], dry_run=dry_run)

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
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print(process.stderr)
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

    reproducibility = CodaLabReproducibility(args.local)
    if args.tune_hyperparameters:
        if args.random_search:
            reproducibility.tune_hyperparameters_random(
                args.worksheet_uuid,
                args.datasets,
                args.algorithm,
                coarse=args.coarse,
                unlabeled_split=args.unlabeled_split,
                dry_run=args.dry_run,
                gpus=args.gpus,
                weak=args.weak,
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
        if args.local:
            reproducibility.output_full_results_local(
                args.experiment, args.path, args.id_val
            )
        else:
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
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to run experiments on (defaults to 1).",
    )
    parser.add_argument(
        "--worksheet-name",
        type=str,
        help="Name of the CodaLab worksheet to reproduce the results on.",
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
        "--weak",
        action="store_true",
        help="Whether to run with weak augmentation for labeled examples (defaults to false).",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Whether to run with coarse-grained domains instead of fine-grained domains (defaults to false).",
    )
    parser.add_argument(
        "--unlabeled-split",
        type=str,
        help="Which unlabeled split to use (defaults to None).",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Whether to just print CodaLab commands instead of running the commands for debugging (defaults to false).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Whether to run locally instead of through worksheets.codalab.org (defaults to false).",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Local path",
    )

    # Parse args and run this script
    args = parser.parse_args()
    main()
