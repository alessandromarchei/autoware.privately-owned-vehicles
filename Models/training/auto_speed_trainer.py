import os
import re
import copy
import csv
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
import yaml
from torch.utils import data

import auto_speed_util as util
from Models.data_utils.load_data_auto_speed import LoadDataAutoSpeed
from Models.model_components.auto_speed_network import AutoSpeedNetwork

from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

import time

def train(args, params, run_dir, log_writer):
    # Model
    model = AutoSpeedNetwork().build_model(version=args.version, num_classes=4)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    current_dir = Path(args.dataset + "/images/training/")
    filenames = [f.as_posix() for f in current_dir.rglob("*") if f.is_file()]

    sampler = None
    dataset = LoadDataAutoSpeed(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=args.workers, pin_memory=True, collate_fn=LoadDataAutoSpeed.collate_fn,
                                persistent_workers=True, prefetch_factor=args.prefetch_factor)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    for epoch in range(args.epochs):
        model.train()
        if args.distributed:
            sampler.set_epoch(epoch)
        if args.epochs - epoch == 10:
            loader.dataset.mosaic = False

        p_bar = enumerate(loader)

        if args.local_rank == 0:
            print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
            p_bar = tqdm.tqdm(p_bar, total=num_steps)

        optimizer.zero_grad()
        avg_box_loss = util.AverageMeter()
        avg_cls_loss = util.AverageMeter()
        avg_dfl_loss = util.AverageMeter()
        # t0 = time.time()
        for i, (samples, targets) in p_bar:
            # t_load = time.time() - t0
            # print("LOAD TIME:", t_load)
            # t0 = time.time()

            step = i + num_steps * epoch
            scheduler.step(step, optimizer)

            samples = samples.cuda().float() / 255

            # Forward
            with torch.amp.autocast('cuda'):
                outputs = model(samples)  # forward
                loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

            avg_box_loss.update(loss_box.item(), samples.size(0))
            avg_cls_loss.update(loss_cls.item(), samples.size(0))
            avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

            loss_box *= args.batch_size  # loss scaled by batch_size
            loss_cls *= args.batch_size  # loss scaled by batch_size
            loss_dfl *= args.batch_size  # loss scaled by batch_size
            loss_box *= args.world_size  # gradient averaged between devices in DDP mode
            loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
            loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

            # Backward
            amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

            # Optimize
            if step % accumulate == 0:
                # amp_scale.unscale_(optimizer)  # unscale gradients
                # util.clip_gradients(model)  # clip gradients
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # torch.cuda.synchronize()

            # Log
            if args.local_rank == 0:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                   avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                p_bar.set_description(s)

        if args.local_rank == 0:
            # mAP
            last = val(args, params, run_dir, ema.ema)

            log_writer.add_scalar("Loss/box", avg_box_loss.avg, epoch + 1)
            log_writer.add_scalar("Loss/cls", avg_cls_loss.avg, epoch + 1)
            log_writer.add_scalar("Loss/dfl", avg_dfl_loss.avg, epoch + 1)

            log_writer.add_scalar("Metrics/mAP", last[0], epoch + 1)
            log_writer.add_scalar("Metrics/mAP@50", last[1], epoch + 1)
            log_writer.add_scalar("Metrics/Recall", last[2], epoch + 1)
            log_writer.add_scalar("Metrics/Precision", last[3], epoch + 1)

            # Update best mAP
            if last[0] > best:
                best = last[0]

            # Save model
            save = {'epoch': epoch + 1,
                    'model': copy.deepcopy(ema.ema)}

            # Save last, best and delete
            torch.save(save, f=f'{run_dir}/weights/last.pt')
            if best == last[0]:
                torch.save(save, f=f'{run_dir}/weights/best.pt')
            del save

    if args.local_rank == 0:
        util.strip_optimizer(f'{run_dir}/weights/best.pt')  # strip optimizers
        util.strip_optimizer(f'{run_dir}/weights/last.pt')  # strip optimizers


@torch.no_grad()
def val(args, params, run_dir, model=None):
    current_dir = Path(args.dataset + "/images/validation/")
    filenames = [f.as_posix() for f in current_dir.rglob("*") if f.is_file()]

    dataset = LoadDataAutoSpeed(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True, collate_fn=LoadDataAutoSpeed.collate_fn, persistent_workers=True,
                             prefetch_factor=args.prefetch_factor)

    plot = False
    if not model:
        plot = True
        model = torch.load(f=f'{run_dir}/weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = AutoSpeedNetwork().build_model(version=args.version, num_classes=4)
    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def get_next_run(path="."):
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Extract numbers from names like run1, run2, ...
    run_ids = []
    for d in subdirs:
        match = re.match(r"run(\d+)$", d)
        if match:
            run_ids.append(int(match.group(1)))

    if not run_ids:
        return 1  # If no runs exist yet, start with run1

    return max(run_ids) + 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', help="dataset directory path")
    parser.add_argument('-c', '--config', default='auto_speed.yaml', type=str, help='yaml file for config')
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--version', default='n', type=str)
    # parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--runs_dir', default="runs", type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--workers', default=8, type=int,help="Number of dataloader workers")
    parser.add_argument('--prefetch_factor', default=2, type=int, help="Number of samples loaded in advance by each worker")


    args = parser.parse_args()

    if args.prefetch_factor == 0:
        args.prefetch_factor = None

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    # Prepare training directory
    if not os.path.exists(args.runs_dir):
        os.makedirs(args.runs_dir)
    next_run = get_next_run(args.runs_dir)
    run_dir = f"{args.runs_dir}/run{next_run}"
    weights_dir = f"{run_dir}/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    log_writer = SummaryWriter(log_dir=run_dir)

    if args.distributed:
        print(f'Running DDP on local rank {args.local_rank}.')
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        print(f'Run directory: {run_dir}')
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open(args.config, errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    train(args, params, run_dir, log_writer)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
