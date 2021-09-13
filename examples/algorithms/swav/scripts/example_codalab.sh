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
dataset="domainnet"

epochs=400
# batch_size=128 # this is per-GPU batch size
batch_size=64
epsilon=0.03  # use throughout
queue_length=3840 # for an effective batch size of 256, this stores the previous 15 batches

# hyperparameters to be tuned
nmb_prototypes=400 # should be 10x the number of classes, this is approx. the number of subpopulations
epoch_queue_starts=500 # based on previous hyperparameter searches, it seems like the queue doesn't help for domainnet

dist_url="env://"

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
# python examples/algorithms/swav/main_swav.py \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/algorithms/swav/main_swav.py \
    --dataset $dataset \
    --dataset_kwargs use_sentry=True source_domain=sketch target_domain=real \
    --root_dir $HOME \
    --log_dir $HOME \
    --nmb_prototypes $nmb_prototypes \
    --queue_length $queue_length \
    --batch_size $batch_size \
    --lr $base_lr \
    --final_lr 0.0006 \
    --seed 31
