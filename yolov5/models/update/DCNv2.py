from models.common import *

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        # 初始化模型的输入通道数（in_channels）、输出通道数（out_channels）、卷积核大小（kernel_size）、
        # 步长（stride）、填充（padding）、膨胀率（dilation）、每组卷积的数量（groups）和变形组的数量（deformable_groups），
        # 并创建权重和偏置张量及偏置批归一化和激活函数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        # 使用 conv_offset_mask 对输入 执行偏移和蒙版（offset and mask）卷积，得到特征偏移量和特征掩码的张量 offset_mask
        offset_mask = self.conv_offset_mask(x)
        # 通过 torch.chunk() 函数将 offset_mask沿着通道维度（dim=1）分成三个张量（o1，o2和mask），
        # 都具有相同的形状，即 [batch, deformable_groups*3, h, w]。（其中，“batch”表示批量大小，“h”和“w”为特征图的高度和宽度）
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        # 将 o1 和 o2 拼接生成偏移张量 offset， mask 进行 sigmoid 激活，得到正确的归一化后的特征变形情况。（此处靠的是可变形卷积的思想）
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # 使用 torch.ops.torchvision.deform_conv2d() 函数对张量 x 进行双向变形卷积。
        # 其中，权重张量为 self.weight，偏移量和掩码张量为前面计算得到的 offset 和 mask，偏置张量为 self.bias，
        # 步幅、填充和膨胀等参数通过 self.stride, self.padding 和 self.dilation 进行设置。
        # 卷积相关组的数量是 self.groups，变形组的数量是 self.deformable_groups
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        # 通过部分归一化、激活函数，返回输出结果 x 作为模型的前向传播计算结果
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        # 重置模型参数： 随机初始化权重系数，将偏置量设为零
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCN(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCNv2(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_DCN(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))