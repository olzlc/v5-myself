import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    # 通过定义特殊方法 repr() 来覆盖默认的输出方式，以便在打印模块对象时更容易理解各个超参数等信息。
    # 此处输出格式为「SimAM(lambda=e_lambda)」，其中 e_lambda 是在构造函数中定义的正则化系数。
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # 获取输入张量的大小（batch size, channel, height, width），然后计算通道方向上均值，并将其用于计算相似度系数。
        b, c, h, w = x.size()
        # 窗口尺寸 w*h 减 1 来计算上下文注意力机制，其中物体的本身占据了一个像素
        n = w * h - 1
        # 使用 x.mean(dim=[2,3], keepdim=True) 计算输入张量沿 H 和 W 维度的平均值
        # 得到一个形状为 (b, c, 1, 1) 的引用值
        # 将这个均值由原始张量中减去，得到了类似于残差结构的结果，对结果进行平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


if __name__ == '__main__':
    input = torch.randn(3, 64, 7, 7)
    model = SimAM()
    outputs = model(input)
    print(outputs.shape)
