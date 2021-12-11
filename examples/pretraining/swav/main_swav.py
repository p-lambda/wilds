# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file has been modified from the original repository's version in the following ways:
# 1. The model loading logic uses a SwAVModel class that acts as a wrapper around WILDS-Unlabeled
#    models.
# 2. The data loading logic uses a CustomSplitMultiCropDataset class that is compatible with all
#    WILDS-Unlabeled datasets.
# More information about both of these classes can be found in the src/ directory.
#

import argparse
import math
import os
import pdb
import shutil
import sys
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
try:
    import apex
    from apex.parallel.LARC import LARC
except ImportError as e:
    print("Apex not found. Proceeding without it...")

try:
    import wandb
except Exception as e:
    print("wandb not found. Proceeding without it...")


import wilds
from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    ParseKwargs,
    plot_experiment,
    populate_defaults_for_swav
)
from src.multicropdataset import CustomSplitMultiCropDataset
from src.model import SwAVModel

from examples.models.initializer import initialize_model
from examples.utils import initialize_wandb

logger = getLogger()
parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
##### dataset params ####
#########################
parser.add_argument('-d', '--dataset', required=True, choices=wilds.unlabeled_datasets)
parser.add_argument('--root_dir', required=True,
                    help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--splits', nargs='+')

#########################
#### data aug params ####
#########################
parser.add_argument("--nmb_crops", type=int, nargs="+", help="list of number of crops")
parser.add_argument("--size_crops", type=int, nargs="+", help="crops resolutions")
parser.add_argument("--min_scale_crops", type=float, nargs="+", help="argument in RandomResizedCrop")
parser.add_argument("--max_scale_crops", type=float, nargs="+", help="argument in RandomResizedCrop")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments (default: [0, 1])")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss (default: 0.1)")
parser.add_argument("--epsilon", default=0.03, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm (default: 0.03)")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", type=int, help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=500,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument("--n_epochs", default=400, type=int,
                    help="number of total epochs to run")
parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs (default: 0)")
parser.add_argument("--batch_size", type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=5005, type=int,
                    help="freeze the prototypes during this many iterations from the start (default: 5005).")
parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--model", type=str, help="convnet architecture. If not set, uses default model specified in WILDS.")
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--checkpoint_freq", type=int, default=50,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--log_dir", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--is_not_slurm_job", type=bool_flag, default=True, help="Set to true if not running in Slurm.")
parser.add_argument("--cpu_only", type=bool_flag, default=False,
                    help="Set to true to run experiment on CPUs instead of GPUs (for debugging).")
parser.add_argument('--pretrained_model_path', default=None, type=str)

# Weights & Biases
parser.add_argument('--use_wandb', type=bool_flag, nargs='?', default=False)
parser.add_argument('--wandb_api_key_path', type=str,
                    help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                    help="Will be passed directly into wandb.init().")

def main():
    global args
    args = parser.parse_args()
    args = populate_defaults_for_swav(args)
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    logger.info(f"Initialized distributed mode and applied WILDS default...\n{args}")

    if args.use_wandb:
        initialize_wandb(args)

    train_dataset = CustomSplitMultiCropDataset(
        args.dataset,
        args.root_dir,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        args,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        **args.loader_kwargs,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    d_out = 1 # this can be arbitrary; final layer is discarded for SwAVModel
    base_model, _ = initialize_model(args, d_out, is_featurizer=True) # discard classifier
    model = SwAVModel(
        base_model, normalize=True, output_dim=args.feat_dim,
        hidden_mlp=args.hidden_mlp, nmb_prototypes=args.nmb_prototypes
    )

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.n_epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.n_epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.log_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.log_dir, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.n_epochs):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.log_dir, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.n_epochs - 1:
                shutil.copyfile(
                    os.path.join(args.log_dir, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

    if args.rank == 0:
        plot_experiment(args.log_dir)


def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": losses.val,
                        "loss_avg": losses.avg,
                    }
                )
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()
