import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        # d_model, max_shape  256 256 256
        print('d_model, max_shape ', d_model, *max_shape)
        pe = torch.zeros((d_model, *max_shape))
        # ([256, 256, 256])
        print('pe shape: ', pe.shape)

        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        # y_position:  torch.Size([1, 256, 256])
        # x_position:  torch.Size([1, 256, 256])
        print('y_position: ', y_position.shape)
        print('x_position: ', x_position.shape)

        if temp_bug_fix:
            # torch.arange(0, d_model//2, 2) 取 0~d_model//2范围内。每隔两个数取一次
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        # div_term 为64个，这里将其扩展为 64x1x1
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        # 4k 4k+1 4K+2 4K+3 故 256个通道，只需要64块

        # div_term:  torch.Size([64, 1, 1])
        print('div_term: ', div_term.shape, div_term[0])

        # print('x_position: ', x_position.shape, x_position[0])
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        # print('pe', pe)
        # print(pe[0,:5,:5]) # sin(x)
        # print(pe[1,:5,:5]) # cos(x)
        # print(pe[2,:5,:5]) # sin(x)
        # print(pe[3,:5,:5]) # cos(x)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        print('add position mebedding')
        print('input x: ', x.shape, x.size(2), x.size(3))
        # print('pe: ', self.pe, self.pe.shape)
        # 这里是对每个像素点都加了位置编码，同理，每个通道也是
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
