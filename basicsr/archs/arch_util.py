import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import init as init
from torch.nn import functional as F
from torch import Tensor, Size
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Optional, List, Tuple, overload, Union
from basicsr.utils import get_root_logger

from einops import rearrange
import numbers


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class LayerNorm(nn.LayerNorm):
    r"""
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_dim = x.ndim
        if x.shape[1] == self.normalized_shape[0] and n_dim > 2:  # channel-first format
            s, u = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / (s + self.eps)
            if self.weight is not None:
                # Using fused operation for performing affine transformation: x = (x * weight) + bias
                n_dim = x.ndim - 2
                new_shape = [1, self.normalized_shape[0]] + [1] * n_dim
                x = torch.addcmul(
                    input=self.bias.reshape(*[new_shape]),
                    value=1.0,
                    tensor1=x,
                    tensor2=self.weight.reshape(*[new_shape]),
                )
            return x
        elif x.shape[-1] == self.normalized_shape[0]:  # channel-last format
            return super().forward(x)
        else:
            raise NotImplementedError(
                "LayerNorm is supported for channel-first and channel-last format only"
            )


class ChannelShuffle(nn.Module):
    def __init__(self):
        super(ChannelShuffle, self).__init__()
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class DownSample(nn.Module):
    r""" 
    DownSample Layer modified from swin-transformer.
    inp_shape: B, C, H, W
    oup_shape: B, 2 * C, H, W

    Args:
        inp_ch (int): Number of input channels.
        norm_type (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, inp_ch, bias, norm_type=LayerNorm):
        super().__init__()
        self.inp_ch = inp_ch
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction = nn.Conv2d(4 * inp_ch, 2 * inp_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = norm_type(4 * inp_ch)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

