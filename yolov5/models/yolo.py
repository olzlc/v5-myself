# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
# 注意力机制
from models.attention.A2Attention import DoubleAttention
from models.attention.BAM import BAMBlock
from models.attention.CBAM import CBAM, C3CBAM
from models.attention.CoordAttention import CoordAtt, C3CA
from models.attention.CoTAttention import CoTAttention
from models.attention.CrissCrossAttention import CrissCrossAttention
from models.attention.ECA import ECAAttention
from models.attention.EffectiveSE import EffectiveSEModule
from models.attention.GAM import GAM_Attention
from models.attention.GC import GlobalContext
from models.attention.GE import GatherExcite
from models.attention.MHSA import MHSA
from models.attention.MobileViTAttention import MobileViTAttention
from models.attention.ParNetAttention import ParNetAttention
from models.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention
from models.attention.S2Attention import S2Attention
from models.attention.SE import SEAttention
from models.attention.SequentialSelfAttention import SequentialPolarizedSelfAttention
from models.attention.SGE import SpatialGroupEnhance
from models.attention.ShuffleAttention import ShuffleAttention
from models.attention.SimAM import SimAM
from models.attention.SK import SKAttention
from models.attention.TripletAttention import TripletAttention
# 可变形卷积层
from models.update.DCNv2 import C3_DCN

from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    # forward: 前向传播的主函数，用于单尺度推理或训练操作。
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # _forward_once: 前向传播的辅助函数，用于执行单个网络层的运算。
    def _forward_once(self, x, profile=False, visualize=False):
        # x为输入的特征图
        y, dt = [], []  # y和dt，用于存储网络输出和性能分析时间
        for m in self.model:
            # 如果当前层m不是来自上一层，则将输入x从之前的层y[m.f]获取，如果m.f是索引则取y[m.f]，否则遍历m.f中的所有元素j
            # 如果j等于-1则赋值x，否则从y[j]中获取
            # 这段代码实现了YOLOv5检测算法中的skip connection功能，即从较浅层通过跨连接的方式将信息传递到较深层。
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 调用_profile_one_layer()函数记录当前层的时间和性能信息。
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 对输入x执行当前层的运算，更新x为运算结果
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # 如果当前层为最后一层，则设置参数"inplace=True"以避免内存泄漏。
        # thop是pytorch自带的计算FLOPs的工具，此处除以10^9表示单位转换成GFLOPs，同时乘以2表示为反向传播的FLOPs数。
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        # 使用time_sync函数计算当前层的执行时间，取10次平均值。
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        # 在日志中记录当前层的模型类型、参数量、运算时间和FLOPs。
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # 记录当前层的总运行时间。
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers，将Conv2d() 和 BatchNorm2d()层融合到一起，减少运算时间和内存占用
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 将batchnorm参数融入卷积核当中
                delattr(m, 'bn')  # 移除batchnorm层
                m.forward = m.forward_fuse  # 使用fuse的forward函数来替代原来的forward函数
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # 打印模型参数
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # 在model对象中对非参数或注册缓存项执行to(), cpu(), cuda(), half()等操作。
        self = super()._apply(fn)
        m = self.model[-1]  # 获取最后一层的模型类型（可能为Detect或Segment）
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    # cfg表示模型的配置文件路径；ch表示输入图片的通道数，默认为RGB三通道；nc表示数据集中目标类别的数量；anchors表示锚框的数量
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # 加载配置文件
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # 模型字典

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 输入通道，在字典中加入ch这个键值对
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 覆盖旧值
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖旧值
        # 搭建网络
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()，取出最后一层预测层
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 通过计算前向传播得到不同尺度下的步长stride和锚框anchors，并将其存储在self.stride和m.anchors属性中
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            # 模型并不知道步长，因此通过一次前馈传播推测出每层步长
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # 检测高低层的描框传入数据时候正确，不能用于大目标的传到低目标中
            check_anchor_order(m)
            # 训练时每层图片不再是一开始规格，因此框也需要同时进行缩小
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            # 还会调用_initialize_biases()函数，对Detect层中的偏置进行初始化操作
            self._initialize_biases()  # only run once

        # 对模型中的权重进行初始化，并打印模型的相关信息和每一层的参数数量
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    # 输入的特征张量x、是否进行多尺度数据增强的标识augment、是否开启分析模式profile和是否可视化输出结果的标识visualize
    # 如果augment为True，则调用_forward_augment函数进行多尺度数据增强后的推理，并返回结果（它与元素数量与_ground_truth_loss返回的结果不同，因此返回None）；
    # 否则，调用_forward_once函数进行单尺度推理。在前向传播期间，开启profile选项将输出更多的中间变量以便于调试等目的，visualize选项则允许输出特定样本的可视化图像。
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 取出键值并赋值
    # anchors表示锚框的尺寸，nc表示类别数，gd和gw分别表示深度和宽度的缩放倍数，act表示激活函数
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # 如果anchors是列表类型，则na等于列表中第一个元素的长度除以2，否则na等于anchors的值
    # 框的数量，在yaml中可以看到[10,13, 16,30, 33,23]有6个值，但是每个都是表示长宽，故/2表示框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    # no为最终输出层数量，每个层级都会有na个层级预测，nc为自己定义类别，而每个类会有自己概率值，其中5表示4+1，即框的坐标信息，而每个框会有自己置信度信息，因此即4+1
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # layers用于存储构建的神经网络层，save用于存储需要保存的层的索引，ch用于存储每一层的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from：-1, number：1, module：'Conv' 卷积层, args:[64,6,2,2]
        m = eval(m) if isinstance(m, str) else m  # 如果是字符串经过eval推断，如果m是一个字符串类型的数据，则对m溯源，源头可以是一个类
        # m被赋值的是一个Conv的字符串：溯源后找到了在common.py文件下定义的一个类，算是完成了一个字符串转变成类的操作
        # 当一行代码要使用变量 x 的值时，Python 会到所有可用的名字空间去查找变量，按照如下顺序:
        # 1）局部命名空间 - 特指当前函数或类的方法。如果函数定义了一个局部变量 x, 或一个参数 x，Python 将使用它，然后停止搜索。
        # 2）全局命名空间 - 特指当前的模块。如果模块定义了一个名为 x 的变量，函数或类，Python 将使用它然后停止搜索。
        # 3）内置命名空间 - 对每个模块都是全局的。作为最后的尝试，Python 将假设 x 是内置函数或变量
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                # 推断a类型
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        # gd为深度倍数，并不是单纯拿yaml中backend的每个值直接放入，还要乘深度倍数
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, C3_DCN}:
            # c1为输入通道，c2为输出通道
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 不是最终输出通道数的话
                c2 = make_divisible(c2 * gw, 8)  # 扩大再变8倍数

            # 拼接变为Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)参数定义部分，方便后续继续使用
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3_DCN}:
                # C3层只有一个参数，需要插入
                args.insert(2, n)  # 重复的数字，将n插到第二个位置
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # 只包含输入通道数(ch[f])，传入上一层的通道数
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # 输出通道数c2就是输入通道数ch[x]的总和，其中x取自f
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            # 在参数列表args的末尾添加一个列表，该列表包含了需要检测或分割的所有输入通道数
            args.append([ch[x] for x in f])
            # 如果参数列表args的第2个元素是整数，则说明需要设置anchor数量，
            # 此时将每个输入通道数的anchor数量设置为整数个（默认每个anchor有2个坐标值）
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # 如果当前层是Segment层，则还需要将参数args[3]除以gw并向下取整，以保证输出通道数是8的倍数
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        # 如果是Contract或Expand类型，则需要根据上一层的通道数和当前层的压缩因子或扩张因子计算当前层的通道数
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2  # 输出通道数c2就是输入通道数ch[f]乘以args[0]的平方
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2  # 输出通道数c2就是输入通道数ch[f]除以args[0]的平方
        # 加入注意力机制后修改
        elif m in {CrissCrossAttention}:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m in {CBAM, C3CBAM, C3CA, CoordAtt}:
            # c1为输入通道，c2为输出通道
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 不是最终输出通道数的话
                c2 = make_divisible(c2 * gw, 8)  # 扩大再变8倍数
            args = [c1, c2, *args[1:]]
            if m in {C3CA}:
                args.insert(2, n)  # 重复的数字，将n插到第二个位置
                n = 1
        # 需要参数的注意力机制
        elif m in {DoubleAttention, BAMBlock, EffectiveSEModule, GAM_Attention, GlobalContext,
                   GatherExcite, MHSA, MobileViTAttention, ParallelPolarizedSelfAttention, ParNetAttention,
                   S2Attention, SEAttention, SequentialPolarizedSelfAttention, ShuffleAttention, SKAttention}:
            args = [ch[f], *args]
        elif m in {CoordAtt, CoTAttention, ECAAttention, SpatialGroupEnhance, SimAM, TripletAttention}:
            c2 = ch[f]
        else:
            c2 = ch[f]  # 如果当前层既不是上述几种类型中的任何一种，那么输出通道数c2就等于输入通道数ch[f]

        # 对于当前层的创建，如果重复次数n大于1，则用nn.Sequential将多个相同类型的层串联起来；否则直接创建一个层
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 加入一些需要保存的层到保存列表
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        # 如果当前元素的索引为0，则表示这是网络的第一层，此时需要将ch清空
        if i == 0:
            ch = []
        # 将当前层的输出通道数加入到ch中，以备下一个元素使用，上层输出通道复用成为当前层输入通道
        ch.append(c2)
    # 返回网络结构和保存列表排序
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s-fire.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否打印模型速度
    parser.add_argument('--profile', action='store_true', default=True, help='profile model speed')
    # 是否打印每层速度
    parser.add_argument('--line-profile', action='store_true', default=True, help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    # 打印模型参数
    model(im, profile=True)
    results = profile(input=im, ops=[model], n=3)
    if not opt.line_profile and not opt.profile:
        if opt.test:
            for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
                try:
                    _ = Model(cfg)
                except Exception as e:
                    print(f'Error in {cfg}: {e}')
        else:  # report fused model summary
            model.fuse()
    # if opt.line_profile:  # profile layer by layer
    #     model(im, profile=True)
    #
    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)
    #
    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')
    # else:  # report fused model summary
    #     model.fuse()
