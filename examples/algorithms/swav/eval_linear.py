# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from unlabeled_extrapolation.datasets.breeds import Breeds, BREEDS_SPLITS_TO_FUNC
BREEDS_DATASETS = BREEDS_SPLITS_TO_FUNC.keys()

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
    ParseKwargs,
    plot_experiment
)
import src.resnet50 as resnet_models

from unlabeled_extrapolation.datasets.breeds import Breeds
from unlabeled_extrapolation.datasets.domainnet import DomainNet

logger = getLogger()

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
# Added by MX
parser.add_argument("--domains", type=str, default=None,
                    help="domain string to pass to dataset")
parser.add_argument("--dataset_name", type=str, default=None,
                    help="name of the dataset")
parser.add_argument('--standardize_ds_size', type=bool_flag, default=False,
                    help='require that all splits use the same size, ' +
                    'specifying which dataset to standardize to')
parser.add_argument('--standardize_to', type=str, default=None,
                    help='Which breeds dataset to which to standardize')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument("--is_not_slurm_job", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    # build data
    target_dataset = None
    if args.dataset_name is None or args.dataset_name == 'imagenet':
        train_dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"))
        if args.standardize_ds_size:
            if args.seed is None:
                raise ValueError('Must provide a seed for downsampling')
            if args.standardize_to is None or args.standardize_to not in BREEDS_DATASETS:
                raise ValueError('Must provide some valid Breeds dataset to standardize to.')
            # calculate size of source and target datasets
            source_ds = Breeds(args.data_path, args.standardize_to, source=True, target=False)
            source_size = len(source_ds)
            target_ds = Breeds(args.data_path, args.standardize_to, source=False, target=True)
            target_size = len(target_ds)
            print(f'Dataset sizes: source ({source_size}), target ({target_size}). '
                  'Standardizing to the smaller size.')
            size_to_use = min(source_size, target_size)
            prng = np.random.RandomState(args.seed)
            permutation = prng.permutation(len(train_dataset.samples))
            train_dataset.samples = [train_dataset.samples[i] for i in permutation[:size_to_use]]
        val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))
        train_dataset.transform = get_train_transform(
            [0.485, 0.456, 0.406], [0.228, 0.224, 0.225]
        )
        val_dataset.transform = get_val_transform(
            [0.485, 0.456, 0.406], [0.228, 0.224, 0.225]
        )
    elif args.dataset_name == 'breeds':
        train_dataset = Breeds(
                args.data_path, split='train',
                source=True, target=False,
                downsample=args.standardize_ds_size, seed=args.seed,
                **args.dataset_kwargs)
        val_dataset = Breeds(
                args.data_path, split='val',
                source=True, target=False,
                **args.dataset_kwargs)
        target_dataset = Breeds(
                args.data_path, split='val',
                source=False, target=True,
                **args.dataset_kwargs)
        train_dataset._transform = get_train_transform(
            train_dataset.means, train_dataset.stds
        )
        val_dataset._transform = get_val_transform(
            train_dataset.means, train_dataset.stds
        )
        target_dataset._transform = get_val_transform(
            train_dataset.means, train_dataset.stds
        )
    elif args.dataset_name == 'domainnet':
        domain_list = args.domains.split(',')
        if len(domain_list) != 2:
            raise ValueError('"domain" param should be of form "source,target"')
        source_domain, target_domain = domain_list
        if args.standardize_ds_size:
            raise ValueError('Dataset standardization not supported for DomainNet.')
        train_dataset = DomainNet(
            source_domain, split='train',
            **args.dataset_kwargs
        )
        val_dataset = DomainNet(
            source_domain, split='test',
            **args.dataset_kwargs
        )
        target_dataset = DomainNet(
            target_domain, split='test',
            **args.dataset_kwargs
        )
        train_dataset._transform = get_train_transform(
            train_dataset.means, train_dataset.stds,
        )
        val_dataset._transform = get_val_transform(
            train_dataset.means, train_dataset.stds,
        )
        target_dataset._transform = get_val_transform(
            train_dataset.means, train_dataset.stds,
        )
    else:
        raise ValueError('Not implemented')

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    if target_dataset is not None:
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        target_loader = None

    column_names = ("epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val")
    if target_dataset is not None:
        column_names += ('loss_tgt', 'prec1_tgt', 'prec5_tgt')

    logger, training_stats = initialize_exp(
        args, *column_names
    )
    logger.info("Building data done")

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
    if args.dataset_name == 'imagenet':
        num_classes = 1000
    else: # TODO: why is this returning wrong number??????
        num_classes = train_dataset.get_num_classes()
        num_classes = 1000
    linear_classifier = RegLog(num_classes, args.arch, args.global_pooling, args.use_bn)

    # convert batch norm layers (if any)
    linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(
        linear_classifier,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=0.9,
        weight_decay=args.wd,
    )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_val_acc": 0., "best_tgt_acc": 0.}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_val_acc = to_restore["best_val_acc"]
    best_tgt_acc = to_restore["best_tgt_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        scores = train(model, linear_classifier, optimizer, train_loader, epoch)
        if args.rank == 0:
            logger.info("Evaluating on validation dataset...")
        scores_val, best_val_acc = validate_network(val_loader, model, linear_classifier, best_val_acc)
        if target_loader is not None:
            if args.rank == 0:
                logger.info("Evaluating on target dataset...")
            scores_tgt, best_tgt_acc = validate_network(target_loader, model, linear_classifier, best_tgt_acc)
            scores_val += scores_tgt
        training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }
            if target_loader is not None:
                save_dict["best_tgt_acc"] = best_tgt_acc
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))

    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 val accuracy: {acc:.1f}".format(acc=best_val_acc))
    if target_loader is not None:
        logger.info("Top-1 tgt accuracy: {acc:.1f}".format(acc=best_tgt_acc))

    if args.rank == 0:
        plot_experiment(args.dump_path)

def get_train_transform(mean, std):
    tr_normalize = transforms.Normalize(
        mean=mean, std=std,
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    return train_transform

def get_val_transform(mean, std):
    tr_normalize = transforms.Normalize(
        mean=mean, std=std,
    )
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    return val_transform

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet18":
                s = 512
            elif arch == "resnet50":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        output = reglog(output)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, linear_classifier, best_acc):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = linear_classifier(model(inp))
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))

    return (losses.avg, top1.avg.item(), top5.avg.item()), best_acc


if __name__ == "__main__":
    main()
