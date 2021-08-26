#!/bin/bash
set -x

# General tips on hyperparameter tuning from the SwAV repo:
#   use a lower epsilon value (--epsilon 0.03 instead of the default 0.05)
#   carefully tune the hyper-parameters
#   freeze the prototypes during first iterations (freeze_prototypes_niters argument)
#   switch to hard assignment
#   remove batch-normalization layer from the projection head
#   reduce the difficulty of the problem (less crops or softer data augmentation)

# Hyperparameter tuning for WILDS
#   Total batch size 256
#   Epochs 400
#   number of prototypes = uniform(10, 20) x number of classes
#   epoch_queue_starts = uniform(60, 200)
#   4096 = queue_length + batch size = 3840 + 256
#   queue_length=3840
#   base_lr=0.6
#   Use default for LR, weight decay and pretty much everything else
#   Update --size_crops 224 96 according to the target resolution

# Fine-tuning
#   Run ERM using the last checkpoint from pre-training (epoch 399)
#   Train with best default hyperparameters from ERM
#   Tune LR and weight decay for ERM with the best SwAV hyperparameters

# Example script to run on the cluster
root_dir="/u/scr/nlp/dro"
log_dir="/u/scr/nlp/dro/swav/test_run"
dataset="domainnet"

epochs=400
batch_size=64 # this is per-GPU batch size
epsilon=0.03  # use throughout
queue_length=3840 # for an effective batch size of 256, this stores the previous 15 batches

# hyperparameters to be tuned
nmb_prototypes=400 # should be 10x the number of classes, this is approx. the number of subpopulations
epoch_queue_starts=500 # based on previous hyperparameter searches, it seems like the queue doesn't help for domainnet

dist_url="tcp://$SLURMD_NODENAME:40001" # TODO: this depends on the specific cluster

# Use linear scaling for learning rate, based on batch size
DEFAULT_LR=4.8
DEFAULT_BATCH_SIZE=4096
NUM_GPUS=4
effective_batch_size=$((batch_size * $NUM_GPUS))
if [ $effective_batch_size = 256 ]; then
    base_lr=0.6
else
    base_lr=$(python3 -c "print($DEFAULT_LR / ($DEFAULT_BATCH_SIZE / $effective_batch_size))")
    if [[ $? -ne 0 ]]; then
        echo 'Error computing batch size, exiting...'
        exit $?
    fi
fi

# If we are not using Slurm, we need to launch with torch.distributed.launch:
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/algorithms/swav/main_swav.py \
python examples/algorithms/swav/main_swav.py \
    --dataset $dataset \
    --dataset_kwargs use_sentry=True source_domain=sketch target_domain=real \
    --root_dir $root_dir \
    --log_dir $log_dir \
    --nmb_crops 2 6 \
    --size_crops 224 96 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.14 \
    --crops_for_assign 0 1 \
    --temperature 0.1 \
    --epsilon $epsilon \
    --sinkhorn_iterations 3 \
    --feat_dim 128 \
    --nmb_prototypes $nmb_prototypes \
    --queue_length $queue_length \
    --epoch_queue_starts $epoch_queue_starts \
    --n_epochs $epochs \
    --warmup_epochs 0 \
    --batch_size $batch_size \
    --lr $base_lr \
    --freeze_prototypes_niters 5005 \
    --weight_decay 0.000001 \
    --loader_kwargs num_workers=4 pin_memory=True drop_last=True \
    --dist_url $dist_url \
    --sync_bn pytorch \
    --is_not_slurm_job false \
    --use_fp16 true \
    --cpu_only false \
    --seed 31