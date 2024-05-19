"""
Codes for paper "AdaNet: A Competitive Adaptive Convolutional Neural Network for Spectral Information Identification".
@article{
    LiSir000,
    title={AdaNet: A Competitive Adaptive Convolutional Neural Network for Spectral Information Identification},
    author={Ziyang Li},
    journal={Pattern Recognition},
    year={2024}
}
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd


# LKA(ResConv) from VAN (https://github.com/Visual-Attention-Network)
class ResConv(nn.Module):
    def __init__(self, dim, k, d, out):
        super(ResConv, self).__init__()
        self.dw_conv = nn.Conv2d(dim, dim, (2 * d - 1, 2 * d - 1), padding=(2 * d - 1) // 2, groups=dim)
        self.dw_d_conv = nn.Conv2d(dim, dim, (k // d, k // d), stride=(1, 1), padding=(((k // d) // 2) * d), groups=dim,
                                   dilation=d)
        self.pw_conv = nn.Conv2d(dim, out, 1)

    def forward(self, x):
        u = x.clone()
        x = self.dw_conv(x)
        x = self.dw_d_conv(x)
        x = self.pw_conv(x)
        u = self.pw_conv(u)
        return u * x


# Get contribution rate
def get_CR(dim=32):
    data_ = pd.read_csv('./dataset/data.csv', header=None)
    data_ = np.array(data_).astype('float64')
    data = data_[:, :520]
    # data = np.random.random((160, 520))
    data = torch.Tensor(data)
    data = data.reshape(-1, 26, 20)
    pool = torch.nn.AdaptiveAvgPool2d(1)
    conv = torch.nn.Conv2d(data.shape[0], dim, 1, 3, 1)

    cr_list = pool(conv(data))
    cr = 0
    sum_cr = sum(cr_list)
    for i in cr_list:
        if i > ((sum_cr / dim) * 1.5):
            cr += 1
    return cr


# AdaConv
class AdaConv(nn.Module):
    def __init__(
            self,
            input_dim=32,
            output_dim=32,
            mode='train',
            cr=1
    ):
        super(AdaConv, self).__init__()
        assert input_dim > 0, 'Input channels must be am integer and more than zero'

        if mode == 'train':
            self.l_chunk = int(get_CR(input_dim))
        else:
            self.l_chunk = cr
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.Sigmoid()
        )
        self.gauss = torch.nn.Parameter(torch.randn(1, input_dim - self.l_chunk, output_dim), requires_grad=True)
        self.scale = input_dim ** -0.5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_el = nn.Conv2d(output_dim, output_dim, kernel_size=(1, 1))

        self.res_conv = ResConv(dim=self.l_chunk, k=3, d=1, out=output_dim)

        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        b, c, h, w = x.size()

        t = self.pool(x).flatten(start_dim=1)
        t = self.mlp(t)

        _, t_max = torch.topk(input=t, k=self.l_chunk, dim=1)
        _, t_min = torch.topk(input=t, k=c - self.l_chunk, dim=1, largest=False)

        effect_more = x[torch.arange(b)[:, None], t_max]
        effect_less = x[torch.arange(b)[:, None], t_min]

        effect_more = self.res_conv(effect_more)

        effect_less_q = rearrange(effect_less, 'b c h w -> b c (h w)')
        effect_less = rearrange(effect_less, 'b c h w -> b (h w) c')
        effect_less_k = effect_less
        corr = torch.bmm(effect_less_q, effect_less_k) * self.scale
        corr = self.softmax(torch.bmm(corr, self.gauss) * self.scale)
        effect_less = torch.bmm(effect_less, corr) * self.scale
        effect_less = rearrange(effect_less, 'b (h w) c -> b c h w', h=h, w=w)
        effect_less = self.conv_el(effect_less)

        mask = effect_more > effect_less
        x = torch.where(mask, 0.9 * effect_more + 0.1 * effect_less, 0.9 * effect_less + 0.1 * effect_more)

        return x


# AdaConv
class GetSelfAttentionMask(nn.Module):
    def __init__(
            self,
            channels=1,
    ):
        super(GetSelfAttentionMask, self).__init__()

        self.to_qk = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(1, 1))
        self.scale = channels ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x):
        qk = self.to_qk(x).chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), qk)
        dots = torch.bmm(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        return attn


class AdaptiveNet(nn.Module):
    def __init__(
            self,
    ):
        super(AdaptiveNet, self).__init__()
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.get_mask = GetSelfAttentionMask(channels=32)
        self.ada_conv = AdaConv(32, 32)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        x = self.stem_conv(x)
        mask = self.get_mask(x)

        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.matmul(mask, x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.ada_conv(x)

        x = self.fc(x)
        return x


# For debug:
if __name__ == '__main__':
    x_in = torch.ones(1, 1, 26, 20)
    model = AdaptiveNet()
    output = model(x_in)
