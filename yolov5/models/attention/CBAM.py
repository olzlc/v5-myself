from models.common import *


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应均值池化层，输出大小为 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化层，输出大小为 1x1
        # 第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放，利用1x1卷积代替全连接
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 1x1 卷积层，用于特征压缩，减少计算复杂度
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 1x1 卷积层，用于特征恢复
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，将数据归一到 [0, 1]

    def forward(self, x):
        # 对输入张量进行自适应均值池化，然后通过 1x1 卷积压缩，并经过 ReLU 激活函数，最终经过一个 1x1 卷积层进行特征恢复
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        # 对输入张量进行自适应最大池化，然后通过 1x1 卷积压缩，并经过 ReLU 激活函数，最终经过一个 1x1 卷积层进行特征恢复
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return


class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16, kernel_size=7):  # ch_in, ch_out, shortcut, groups, expansion
        super(CBAMBottleneck,self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层：输入通道数为c1，输出通道数为c_，卷积核大小为1，步幅为1
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 第二个卷积层：输入通道数为c_，输出通道数为c2，卷积核大小为3，步幅为1，分组进一步卷积
        self.add = shortcut and c1 == c2  # 定义是否使用捷径跨越操作
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        #self.cbam=CBAM(c1,c2,ratio,kernel_size)
    def forward(self, x):
        # 先使用第一个卷积层对输入张量x做处理，再使用第二个卷积层进行进一步的卷积操作
        x1 = self.cv2(self.cv1(x))
        out = self.channel_attention(x1) * x1  # 使用通道注意力机制和元素级别的乘法操作
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return x + out if self.add else out

class C3CBAM(C3):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut) for _ in range(n)))
