"""
Helper code to run multiple iterations of Noisy Student, using the same hyperparameters between iterations. The initial teacher's weights must be provided by the command line.

Normally, to run 2 warm-started iterations with some initial teacher weights, one would run a sequence of commands:
    python examples/run_expt.py --root_dir $HOME --log_dir ./student1 --dataset DATASET --algorithm NoisyStudent --unlabeled_split test_unlabeled --teacher_model_path teacher_weights.pth --pretrained_model_path teacher_weights.pth
    python examples/run_expt.py --root_dir $HOME --log_dir ./student2 --dataset DATASET --algorithm NoisyStudent --unlabeled_split test_unlabeled --teacher_model_path ./student1/model.pth --pretrained_model_path ./student1/model.pth

With this script, to run 2 warm-started iterations with some initial teacher weights:
    python examples/noisy_student_wrapper.py 2 teacher_weights.pth --root_dir $HOME --log_dir . --dataset DATASET --unlabeled_split test_unlabeled

i.e. usage:
    python examples/noisy_student_wrapper.py [NUM_ITERS] [INITIAL_TEACHER_WEIGHTS] [REST OF RUN_EXPT COMMAND STRING]

Notes:
    - Students are all warm-started with the current teacher's weights.
    - This command will use the FIRST occurrence of --log_dir (instead of the last).
"""
import argparse
import os
import pathlib
import pdb
import subprocess

SUCCESS_RETURN_CODE = 0

parser = argparse.ArgumentParser()
parser.add_argument("num_iters", type=int)
parser.add_argument("initial_teacher_path", type=str)  # required
parser.add_argument("cmd", nargs=argparse.REMAINDER)
args = parser.parse_args()

assert args.initial_teacher_path.endswith(".pth")
assert os.path.exists(
    args.initial_teacher_path
), f"Model weights did not exist at {args.initial_teacher_path}"
prefix = pathlib.Path(__file__).parent.resolve()


def remove_arg(args, arg_to_remove):
    idx = args.cmd.index(f"--{arg_to_remove}")
    value = args.cmd[idx + 1]
    args.cmd = args.cmd[:idx] + args.cmd[idx + 2 :]
    return value


# Parse out a few args that we need
try:
    idx = args.cmd.index("--log_dir")
    log_dir = args.cmd[idx + 1]
    args.cmd = (
        args.cmd[:idx] + args.cmd[idx + 2 :]
    )  # will need to modify this between iters, so remove from args.cmd
except:
    log_dir = "./logs"  # default in run_expt.py

idx = args.cmd.index("--dataset")
dataset = args.cmd[idx + 1]

try:
    idx = args.cmd.index("--seed")
    seed = args.cmd[idx + 1]
except:
    seed = 0  # default in run_expt.py

try:
    idx = args.cmd.index("--dataset_kwargs")
    fold = args.cmd[idx + 1]
    assert fold.startswith("fold=")
    fold = fold.replace("fold=", "")
except:
    fold = "A"

# Train the teacher model without unlabeled data and default values for gradient_accumulation_steps and n_epochs
unlabeled_split = remove_arg(args, "unlabeled_split")
gradient_accumulation_steps = remove_arg(args, "gradient_accumulation_steps")
n_epochs = remove_arg(args, "n_epochs")

# Run student iterations
for i in range(1, args.num_iters + 1):
    if i == 1:
        teacher_weights = args.initial_teacher_path
    else:
        if dataset == "poverty":
            teacher_weights = (
                f"{log_dir}/student{i - 1}/{dataset}_fold:{fold}_epoch:best_model.pth"
            )
        else:
            teacher_weights = (
                f"{log_dir}/student{i-1}/{dataset}_seed:{seed}_epoch:best_model.pth"
            )
    cmd = (
        f"python {prefix}/run_expt.py --algorithm NoisyStudent {' '.join(args.cmd)}"
        + f" --unlabeled_split {unlabeled_split} --gradient_accumulation_steps {gradient_accumulation_steps}"
        + f" --n_epochs {n_epochs} --log_dir {log_dir}/student{i}"
        + f" --teacher_model_path {teacher_weights}"
        + f" --pretrained_model_path {teacher_weights}"  # warm starting
    )
    print(f">>> Running {cmd}")
    return_code = subprocess.Popen(cmd, shell=True).wait()
    if return_code != SUCCESS_RETURN_CODE:
        raise RuntimeError(
            f"FAILED: Iteration {i} failed with return code: {return_code}"
        )

print(">>> Done!")
