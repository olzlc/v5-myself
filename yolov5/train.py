# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
# 获取本地进程的 rank，如果没有设置则默认为 -1。
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 获取全局进程的 rank，如果没有设置则默认为 -1
RANK = int(os.getenv('RANK', -1))
# 获取全局进程总数，如果没有设置则默认为 1
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# 获取代码仓库的 Git 信息
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 如果是字符串将其解析为字典类型
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # 将opt.hyp设置为超参数字典的副本，以便在保存检查点时使用

    # 保存模型的超参数和训练选项到对应的yaml文件中
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 创建了一个 Loggers 的实例并且通过 callbacks 注册了这个实例的方法作为回调函数，以在训练过程中记录日志和保存模型等信息
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # 如果正在从远程数据集下载数据，则可以通过 loggers.remote_dataset 属性来获取该数据集的字典
        data_dict = loggers.remote_dataset
        if resume:  # 如果正在从检查点恢复运行，则会将四个参数的值从检查点中加载
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # 初始化随机种子，确保每次运行结果相同，可复现
    # torch_distributed_zero_first是一个上下文管理器，用于在PyTorch分布式训练中确保某些操作只在rank=0的进程上执行一次
    # torch_distributed_zero_first将在rank=0的进程上执行上下文管理器内的代码，而其他进程将跳过该代码块
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # 如果是单类别检测，则类别数为 1，否则获取数据集中的类别数
    # 如果是单类别检测，且类别名称数量不为 1，则类别名称为数据集中的类别名称，否则为data_dict['names']
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # 如果验证集为 COCO 数据集，则 is_coco 为 True
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    # 如果weights是以.pt为后缀的字符串，则会尝试从本地或者远程下载对应的预训练权重文件
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # 本地没找到，下载
        # 将权重加载到CPU上，并根据配置文件或者预设的anchors创建模型
        ckpt = torch.load(weights, map_location='cpu')  # 从本地文件系统加载预训练权重文件 weights，将其加载到 CPU 内存上，以避免 CUDA 内存泄漏问题
        # 新建模型，虽然上面传入模型，但是最后nc类别数并不是模型训练那样，也不同类，后面将模型参数迁移并训练，因为预训练可能对新类别学习有帮助
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 根据训练配置 hyp 和网络配置 cfg 中的设定，创建需要剔除的参数键名列表 exclude
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # 将模型参数的类型转换为 FP32 格式，存储在新的字典对象 csd 中，以匹配当前模型的参数类型
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 取出 ckpt 中与当前模型参数键名交集的参数值，存储在 csd 中
        model.load_state_dict(csd, strict=False)  # 将 csd 中的参数值加载到当前模型中，参数 strict=False 表示可以允许不完全匹配的情况，即跳过部分键名不匹配的参数
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 如果没有预训练权重，则直接根据配置文件创建模型
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    # 检查模型是否支持混合精度训练（AMP）
    amp = check_amp(model)  # check AMP

    # 用户在训练 YOLOv5 模型时选择需要冻结的层以提高训练效果
    # 如果 freeze 列表长度大于 1，则使用列表中的值作为层数；否则，使用 freeze[0] 作为层数
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # 通过遍历模型中的所有参数，并将 requires_grad 设置为 True，来训练所有层
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 如果参数名包含在 freeze 列表中，则将 requires_grad 设置为 False，从而冻结该层。
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # 设置模型的最大步长
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 检查图像大小，以确保它是网格大小的倍数

    # 基于模型、图像大小和自动混合精度（amp）设置来估计训练的最佳批量大小
    if RANK == -1 and batch_size == -1:  # 如果在单个GPU上运行，它将自动估计最佳批处理大小
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # 优化器
    nbs = 64  # 优化器的批量大小为64
    accumulate = max(round(nbs / batch_size), 1)  # 表示在优化之前累积损失的批量大小
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 权重衰减的缩放因子
    # 创建优化器对象
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 学习率调度器
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']，使用余弦退火调度器
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear 线性调度器
    # 使用 `LambdaLR` 类创建一个学习率调度器对象，并传入优化器和 `lf` 函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    # 如果是主进程，即 RANK 为 -1 或 0，则创建一个 ModelEMA 类的实例对象 ema
    # EMA)技术是一种常见的优化方法，用于平滑计算中的变化，尤其是在深度学习中，常用于模型权重更新的平滑，以减少训练过程中的抖动
    # 在深度学习中，通常采用随机梯度下降 (SGD) 或其变体作为模型训练的优化算法。在使用这些算法进行模型训练时，模型参数会在每个批次或每个 epoch 中被更新。然而，这样的频繁更新可能会导致模型参数波动或震荡，从而影响模型的性能。
    # EMA 技术通过对模型参数的移动平均来减轻这种波动和震荡，使得模型的更新更加平滑
    # EMA 维护了一个指数加权平均 (exponentially weighted average) 的缓存，用于记录每个参数的历史变化趋势
    # 在更新模型参数时，EMA 会同时更新指数加权平均的值，以更加平滑地更新模型参数
    # 通过这种方式，EMA 技术可以帮助优化算法更好地收敛，从而提高模型的性能
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    # 初始化最佳精度和开始的训练 epoch
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            # 调用 smart_resume 函数来执行从断点处继续训练的逻辑，并将返回的 best_fitness、start_epoch 和 epochs 分别赋值给变量
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # 多卡才需要DP和SyncBatchNorm
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        # 当迭代次数或者epoch足够大的时候，用nn.DataParallel函数来用多个GPU来加速训练
        # DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        # 将模型中的torch.nn.BatchNormND层转换为torch.nn.SyncBatchNorm层。
        # 引入SyncBN，跟一般所说的普通BN的不同在于工程实现方式：SyncBN能够完美支持多卡训练，而普通BN在多卡模式下实际上就是单卡模式
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    # dataloader理解为数据抓取器
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    # 将所有样本的标签拼接到一起shape为(total, 5)，统计后做可视化
    # 获取标签中最大的类别值，并于类别数作比较
    # 如果大于类别数则表示有问题
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            # 如果不是恢复训练，则检查是否需要自动设置anchor，并调用 check_anchors() 函数进行设置。
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision 将模型的anchor参数的精度降低至FP16，以减少内存消耗
        # 用于执行训练前的准备工作，例如对标签和类别名称的处理
        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # 将一些模型属性和超参数添加到模型
    nl = de_parallel(model).model[-1].nl  # 检测层的数量（nl），用于在超参数（hyp），使它们适应不同的检测层
    # 下面三个超参数含义为损失函数因子
    hyp['box'] *= 3 / nl  # 缩放目标框（box）
    hyp['cls'] *= nc / 80 * 3 / nl  # 类别预测（cls）
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers,物体置信度（obj）的权重
    hyp['label_smoothing'] = opt.label_smoothing  # 标签平滑参数（label_smoothing），这是一个用于减轻过拟合的技巧，它将真实标签值向其他类别的概率分布平滑化
    model.nc = nc  # 模型的类别数量（nc），将其附加到模型中
    model.hyp = hyp  # attach hyperparameters to model
    # 每个类别的权重（class_weights），用于计算损失函数时平衡类别之间的贡献
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1  # 上次更新参数计数器值
    maps = np.zeros(nc)  # mAP per class，训练过程中map值
    # 结果
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 自动回归精度训练
    stopper, stop = EarlyStopping(patience=opt.patience), False  # 提前终止
    compute_loss = ComputeLoss(model)  # 初始损失函数
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # 训练时由于图片识别目标可能识别比较容易，有的可能比较困难，因此可以给模型分配采样权重，使模型额外关注难识别样本
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            # 某一类数量权重
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重，增大类别权重可以增大难采样部分被识别概率
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 类别权重换算到图片维度，即每张图片采样权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 增加权重后再随机重采样，后面一批批分析图片中，难识别样本数会增加

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # 初始损失值，框回归损失，类别损失和置信度损失，3种
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        # 展示训练进度
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()  # 梯度归零
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # 热身，前面几次学习率不那么大
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # 预测框和标注框直接损失值
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 多批数据进行累积，统一进行更新
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # 将优化器中的梯度进行反归一化处理
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 对模型参数的梯度进行裁剪，防止梯度爆炸
                scaler.step(optimizer)  # 使用优化器对模型参数进行更新
                scaler.update()  # 更新scaler，以便下一次使用
                optimizer.zero_grad()  # 清空优化器中的梯度信息
                if ema:
                    ema.update(model)  # 模型进行指数移动平均处理
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # 在loggers/_init_.py中有同名函数，调用
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # 更新学习率
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # ema添加属性
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断是否最终一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                # 不是最后一轮，在验证集再跑一遍
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            # 计算拟合度，对多个指标加权求和
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            # 判断最好，是否记录下来
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # 判断拟合度是否都没上升，是否提前训练结束，
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 权重文件路径
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
    # 存储模型结构的配置文件，指定了一些参数信息和backbone的结构信息
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s-fire.yaml', help='model.yaml path')
    # 存储训练、测试数据的文件
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire-smoke.yaml', help='dataset.yaml path')
    # 模型的超参数路径
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # epoch、batchsize、iteration三者之间的联系
    # 1、batchsize是批次大小，假如取batchsize=24，则表示每次训练时在训练集中取24个训练样本进行训练。
    # 2、iteration是迭代次数，1个iteration就等于一次使用24（batchsize大小）个样本进行训练。
    # 3、epoch：1个epoch就等于使用训练集中全部样本训练1次
    # 训练过程中整个数据集将被迭代多少次
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    # 一次看完多少张图片才进行权重更新，梯度下降的mini-batch
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    # 输入图片宽高
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    # parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # 进行矩形训练
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # 断点续训：即是否在之前训练的一个模型基础上继续训练，default 值默认是 False；
    # 如果想采用断点续训的方式，推荐将 default=False 改为 default=True。
    # 随后在终端中键入如下指令：
    # python train.py --resume D:\Pycharm_Projects\yolov5-6.1-4_23\runs\train\exp19\weights\last.pt上一次中断时保存的pt文件路径
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # 仅保存最终checkpoint
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # 仅验证最后一个 epoch，而不是每个 epoch 都进行验证
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # 禁用自动锚框（AutoAnchor）的功能，自动锚点的好处是可以简化训练过程
    # 自动锚定框选项，训练开始前，会自动计算数据集标注信息针对默认锚定框的最佳召回率，当最佳召回率大于等于0.98时，则不需要更新锚定框；
    # 如果最佳召回率小于0.98，则需要重新计算符合此数据集的锚定框
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # 禁止保存绘图文件
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # 多少代进化一次超参数
    # yolov5使用遗传超参数进化，提供的默认参数是通过在COCO数据集上使用超参数进化得来的。由于超参数进化会耗费大量的资源和时间，所以建议大家不要动这个参数
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # 指定一个 Google Cloud Storage 存储桶
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # 指定图像的缓存类型，参数未指定时的默认值为 'ram'
    # 是否提前缓存图片到内存，以加快训练速度，默认False；开启这个参数就会对图片进行缓存，从而更好的训练模型
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # 指定在训练时使用加权图像选择
    # 是否启用加权图像策略，默认是不开启的；主要是为了解决样本不平衡问题；开启后会对于上一轮训练效果不好的图片，在下一轮中增加一些权重
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # cuda设备, i.e. 0 or 0,1,2,3 or cpu
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 多尺度训练，img-size +/- 50%
    # 是否启用多尺度训练，默认是不开启的；多尺度训练是指设置几种不同的图片输入尺度，训练时每隔一定iterations随机选取一种尺度训练，这样训练出来的模型鲁棒性更强
    # 多尺度训练在比赛中经常可以看到他身影，是被证明了有效提高性能的方式。输入图片的尺寸对检测模型的性能影响很大，在基础网络部分常常会生成比原图小数十倍的特征图，导致小物体的特征描述不容易被检测网络捕捉。通过输入更大、更多尺寸的图片进行训练，能够在一定程度上提高检测模型对物体大小的鲁棒性。
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # 单类别的训练集，单类别还是多类别；默认为False多类别
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # 指定优化器的类型，选择优化器；默认为SGD，可选SGD，Adam，AdamW 。
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # 启用同步批规范化，只在分布式数据并行（DDP）模式下可用，是否开启跨卡同步BN；开启参数后即可使用SyncBatchNorm多 GPU 进行分布式训练
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # 指定数据加载器的最大工作进程数，仅在分布式数据并行（DDP）模式下有用
    # 这里经常出问题，Windows系统报错时可以设置成0
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (per RANK in DDP mode)')
    # 指定保存训练结果的路径
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # 重命名文件名
    parser.add_argument('--name', default='exp', help='save to project/name')
    #允许使用现有的project/name，不增加计数器
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 使用四路数据加载器，官方发布的开启这个功能后的实际效果，好处是在比默认640大的数据集上训练效果更好，副作用是在640大小的数据集上训练效果可能会差一些
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # 使用余弦退火学习率调度器
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # 标签平滑的epsilon值，是否对标签进行平滑处理，默认是不启用的
    # 在训练样本中，我们并不能保证所有sample都标注正确，如果某个样本标注错误，就可能产生负面印象，如果我们有办法“告诉”模型，样本的标签不一定正确，
    # 那么训练出来的模型对于少量的样本错误就会有“免疫力”采用随机化的标签作为训练数据时，损失函数有1-ε的概率与上面的式子相同，
    # 比如说告诉模型只有0.95概率是那个标签
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # EarlyStopping的耐心度（没有改进的时期数），如果模型在default值轮数里没有提升，则停止训练模型
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # 冻结层次数量，如指定backbone = 10，first3 = 0、1、2，可以在yolov5s.yaml中查看主干网络层数
    # 冻结训练是迁移学习常用的方法，当我们在使用数据量不足的情况下，通常我们会选择公共数据集提供权重作为预训练权重，
    # 我们知道网络的backbone主要是用来提取特征用的，一般大型数据集训练好的权重主干特征提取能力是比较强的，
    # 这个时候我们只需要冻结主干网络，fine-tune后面层就可以了，不需要从头开始训练，大大减少了实践而且还提高了性能
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # 每x个epoch保存一次检查点（如果小于1，则禁用），用于设置多少个epoch保存一下checkpoint；
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # 全局训练种子
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    # 自动 DDP 多 GPU 参数，不要修改，单GPU设备不需要设置
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    # 实体名称，在线可视化工具，类似于tensorboard
    parser.add_argument('--entity', default=None, help='Entity')
    # 上传数据集。如果指定了 'val' 选项，则仅上传验证数据集
    # 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    # 设置记录边界框图像的间隔
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    # 要使用的数据集工件的版本名称。默认为 'latest'，这个功能作者还未实现
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')
    # 在解析过程中遇到不认识的参数，parse_args() 函数会抛出异常，而 parse_known_args() 函数则会将不认识的参数存储在一个列表中返回，同时返回已知参数的值
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # 用于恢复模型训练
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 如果这个参数是一个字符串类型，那么就调用check_file()函数检查这个文件是否存在，如果不存在就会抛出异常。
        # 如果这个参数是None或者其他非字符串类型，那么就调用get_latest_run()函数获取最新的checkpoint文件。
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                # 将 opt.yaml 文件解析为一个 Python 字典
                d = yaml.safe_load(f)
        else:
            # 从指定的last.pt文件中加载训练期间的各种参数和选项，map_location指定了加载对象的位置，这里指定为CPU
            d = torch.load(last, map_location='cpu')['opt']
        # 将字典转换为命名空间，将新的配置参数命名空间赋值给 opt 变量，后面可以继续使用这些参数进行训练了
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # 获取该URL所指向的本地文件路径，以避免HUB恢复鉴权超时
    else:
        # 检查指定的数据集、配置文件、超参数文件和权重文件是否存在并进行必要的初始化
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # Path对象表示文件路径。.stem是这个对象的一个属性，它返回路径的最后一部分，并且不包括文件扩展名
        # 初始化为指定的项目名称和模型名称组成的路径，并使用 increment_path 函数在该路径上进行必要的初始化（例如，如果该路径已经存在，则会自动加上后缀 _1、_2 等）
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        # 两个选项不支持多GPU模式
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        # 判断是否设置了--batch-size -1选项，若设置了则需要传入一个有效的--batch-size选项
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        # 判断--batch-size是否是WORLD_SIZE的倍数
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        # 如果当前机器上的GPU数量大于LOCAL_RANK，则会将当前设备的编号设置为LOCAL_RANK
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # 设置当前进程使用的 GPU ID
        torch.cuda.set_device(LOCAL_RANK)
        # 将当前进程的设备类型设置为 CUDA，同时指定了 GPU ID
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化了进程组，根据系统的不同，支持 nccl 或者 gloo 的方式来实现进程通信，以实现多GPU训练
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        # nccl （NVIDIA Collective Communications Library）是一种面向多 GPU 的通信库，
        # 它是 NVIDIA 提供的高效的集体通信库，可以用于快速的数据并行和模型并行。nccl 通过 NVIDIA GPU 上的硬件加速网络实现了高效的多 GPU 通信。
        # gloo 则是一种多节点的 CPU/GPU 通信后端，它使用了类似于 MapReduce 的思路来实现数据并行和模型并行。
        # gloo 的实现不依赖于硬件特性，因此可以在大多数平台上使用。

    # 训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # 演化超参数（可选）
    else:
        # 超参数进化元数据（变异尺度0-1、下限、上限）
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        # 打开用户提供的超参数文件(opt.hyp)
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载为一个Python字典对象，以便稍后在模型训练过程中使用
            # 如果该文件中没有指定锚框(anchor boxes)的数量，那么默认设置为3个
            if 'anchors' not in hyp:
                hyp['anchors'] = 3
        # 表示不自动生成anchors
        if opt.noautoanchor:
            # 需要从超参数字典hyp和元数据字典meta中将anchors项删除
            del hyp['anchors'], meta['anchors']
        # 将它们设置为True时，表示在训练过程中只验证并保存最终epoch的模型
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei指示hyp字典中哪些值是可进化的，通过迭代hyp字典中的值并使用isinstance（）函数检查每个值是否是int或float的实例来创建ei。
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # 下载一个名为"evolve.csv"的文件到本地的指定目录中
            # 通过运行系统命令来执行此操作，gsutil命令用于访问和管理 GCS 上的对象，cp参数用于复制文件。如果本地目录下已经有相同的文件，将会被覆盖
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv)])

        # 进行YOLOv5的超参数自动优化，即进行进化算法，生成多个新的超参数组合并训练多个模型，根据这些模型的表现选择出表现最好的超参数组合，并继续进化。
        # 其中，循环次数为 opt.evolve 次，每次进化都会生成新的超参数组合
        for _ in range(opt.evolve):
            if evolve_csv.exists():  # 如果evolve.csv存在：选择最佳超参数并进行变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single' 或者 'weighted'
                # 从 evolve.csv 文件中读取已有的超参数和它们的评估结果
                # 跳过第一行（因为第一行是列名），将剩余的行作为二维数组返回，每一列代表一个超参数的值，每一行代表一个超参数组合的评估结果
                # 在这段代码中，ndmin=2 意味着返回的数组至少是二维的，即使只有一行或一列，这使得代码更加健壮
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # 考虑的以前结果数量，从之前的测试结果中最多选择5个进行进化
                x = x[np.argsort(-fitness(x))][:n]  # 最高n个变异，按照测试性能从高到低排序，并返回下标
                # 计算每个测试结果的适应度，然后对适应度进行归一化处理，确保所有的权重值都大于0，加上了一个很小的数（1E-6）来避免权重出现0的情况
                w = fitness(x) - fitness(x).min() + 1E-6  # 权重（总和 > 0）
                # 如果parent等于'single'或者前几代中只有一个结果，就使用随机选择的方法从前几代的结果中选择一个作为父代
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    # 使用加权平均的方法选择父代，其中每个前几代的结果都按照其适应度值作为权重，加权平均得到一个新的参数组合
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # 对超参数的变异操作
                mp, s = 0.8, 0.2  # mp是变异概率，s是变异的幅度
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # 当前超参数值在0到1之间的增益因子
                ng = len(meta)
                v = np.ones(ng)  # v是当前的变异向量，初始值都是1。在变异过程中，程序会在每个超参数上分别进行变异
                while all(v == 1):  # 变异直到发生变化（防止重复）
                    # 判断是否进行变异的概率是否大于随机生成的一个0到1之间的随机数，
                    # 然后根据变异的概率和幅度生成一个随机数作为当前超参数的变异因子，最后将当前超参数乘以变异因子进行变异
                    # 如果变异后的超参数不在0.3到3之间，程序会将其剪切到这个区间内
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将经过选定的变异策略后的新参数应用到模型的超参数上
                # 基因会根据概率和标准差进行变异，得到一个新的基因值，然后将这个新的基因值乘以一个变异因子 v[i]，将得到的结果更新到超参数 hyp 的对应键 k 上
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # 将每个参数的取值限制在其允许的范围内，并将其四舍五入到小数点后五位
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 低于其规定的下限，则将其设置为其下限
                hyp[k] = min(hyp[k], v[2])  # 高于其规定的上限，则将其设置为其上限
                hyp[k] = round(hyp[k], 5)  # 将每个参数的值舍入到小数点后五位

            # 训练
            # 首先，将当前的超参数（hyp）复制一份，以免对原始超参数进行更改。然后使用当前超参数进行训练，将训练结果存储在results变量中。
            results = train(hyp.copy(), opt, device, callbacks)
            # 创建一个空的Callbacks对象，以确保不会使用上一次的回调
            callbacks = Callbacks()
            # 调用print_mutation()函数，将关键指标（precision、recall、mAP等）的值、超参数和保存目录作为参数传递，以将结果写入日志文件中，用于后续分析和参考
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # 用于绘制超参数进化过程中的结果，并输出训练的结果保存路径和使用的命令行参数示例
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
