#!/bin/bash
set -x

root_dir="/u/scr/nlp/dro"
log_dir="/u/scr/nlp/dro/swav/test_run"
# root_dir="../data"
# log_dir="../logs"

arch="resnet50"
batch_size=128 # this is per-GPU batch size

# Tips on hyperparameter tuning from the SwAV repo:
# use a lower epsilon value (--epsilon 0.03 instead of the default 0.05)
# carefully tune the hyper-parameters
# freeze the prototypes during first iterations (freeze_prototypes_niters argument)
# switch to hard assignment
# remove batch-normalization layer from the projection head
# reduce the difficulty of the problem (less crops or softer data augmentation)

# hyperparameters to be tuned
epsilon=0.03
nmb_prototypes=3000 # should be 10x the number of classes, this is approx. the number of subpopulations
queue_length=3840 # for an effective batch size of 256, this stores the previous 15 batches
epoch_queue_starts=500 # based on previous hyperparameter searches, it seems like the queue doesn't help for domainnet
epochs=400
# TODO: what about LR and weight decay? -Tony
# What about cropping? -Tony

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
# TODO: why / 1000? do we need to tune this too? -Tony
final_lr=$(python3 -c "print($base_lr / 1000)")
echo "Using base_lr=$base_lr and final_lr=$final_lr"

# If we are not using Slurm, we neeed to launch with torch.distributed.launch. Ex:
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/algorithms/swav/main_swav.py \
python examples/algorithms/swav/main_swav.py \
    --dataset domainnet \
    --root_dir $root_dir \
    --dataset_kwargs source_domain=sketch target_domain=real \
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
    --final_lr $final_lr \
    --freeze_prototypes_niters 5005 \
    --weight_decay 0.000001 \
    --loader_kwargs num_workers=4 pin_memory=True drop_last=True \
    --dist_url $dist_url \
    --model $arch \
    --sync_bn pytorch \
    --log_dir $log_dir \
    --rank 0 \
    --world_size 1 \
    --is_not_slurm_job false \
    --use_fp16 true \
    --cpu_only false \


# WARNING: below here hasn't been test before, pretrained checkpoint should be dynamically calculated
#python examples/algorithms/swav/eval_semisup.py \
#    --dataset domainnet \
#    --root_dir $root_dir \
#    --source real \
#    --target sketch \
#    --log_dir $log_dir \
#    --arch $arch \
#    --pretrained $log_dir/checkpoints/ckp-399.pth \
#    --dist_url $dist_url