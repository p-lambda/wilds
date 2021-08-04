#!/bin/bash

set -x

root_dir="/u/scr/nlp/dro"
domains="real,sketch"

arch="resnet50"
batch_size=128 # this is per-GPU batch size

# hyperparameters to be tuned
epsilon=0.03
nmb_prototypes=3000 # should be 10x the number of classes, this is approx. the number of subpopulations
queue_length=3840 # for an effective batch size of 256, this stores the previous 15 batches
epoch_queue_starts=500 # based on previous hyperparameter searches, it seems like the queue doesn't help for domainnet
epochs=400

# first, make sure that the dump-path is named as we'd like, then create that directory (o.w. it will fail)
dump_path=examples/algorithms/swav/checkpoints/domainnet-real-sketch
dist_url="tcp://$SLURMD_NODENAME:40001" # TODO: this depends on the specific cluster

# Use linear scaling for learning rate, based on batch size
DEFAULT_LR=4.8
DEFAULT_BATCH_SIZE=4096
NUM_GPUS=1 # TODO: automatically detect number of gpus
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
final_lr=$(python3 -c "print($base_lr / 1000)")
echo "Using base_lr=$base_lr and final_lr=$final_lr"

python examples/algorithms/swav/main_swav.py \
    --dataset domainnet \
    --root_dir $root_dir \
    --dataset_kwargs domains=$domains \
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
    --epochs $epochs \
    --batch_size $batch_size \
    --base_lr $base_lr \
    --final_lr $final_lr \
    --freeze_prototypes_niters 5005 \
    --wd 0.000001 \
    --warmup_epochs 0 \
    --workers 4 \
    --dist_url $dist_url \
    --arch $arch \
    --use_fp16 true \
    --sync_bn pytorch \
    --dump_path $dump_path


# WARNING: below here hasn't been test before, pretrained checkpoint should be dynamically calculated
python examples/algorithms/swav/eval_semisup.py \
    --dataset domainnet \
    --root_dir $root_dir \
    --source real \
    --target sketch \
    --dump_path $dump_path \
    --arch $arch \
    --pretrained $dump_path/checkpoints/ckp-399.pth \
    --dist_url $dist_url