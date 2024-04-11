# -*- encoding: utf-8 -*-
'''
@File    :   reformer_arch.py
@Time    :   2023/12/12 10:42:58
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, List, Sequence
from einops import rearrange
from basicsr.archs.arch_util import LayerNorm, DownSample, make_divisible
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.local_arch import Local_Base
from mmcv.ops import ModulatedDeformConv2d
import math
import numpy as np
import time

 
class SimpleGate(nn.Module):
    def __init__(self):
        super(SimpleGate, self).__init__()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
    
class SimpleSeModule(nn.Module):
    def __init__(self, inp_chan: int):
        super(SimpleSeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_chan, inp_chan, kernel_size=1, bias=True),
        )

    def forward(self, x: Tensor):
        return x * self.se(x)
    

class SeModule(nn.Module):
    def __init__(self, inp_chan, reduction=4):
        """
        Channel Attention aplied in MobileNetv3
        """
        super(SeModule, self).__init__()
        expand_size =  max(inp_chan // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_chan, expand_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, inp_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class HFBlock(nn.Module):
    def __init__(self, 
                 inp_chan: int, 
                 bias: bool):
        super(HFBlock, self).__init__()
        self.conv = nn.Conv2d(inp_chan, inp_chan, kernel_size=1, bias=bias)
        self.down = nn.AvgPool2d(2)
    def forward(self, x: Tensor):
        x1 = self.conv(x)
        x2 = self.down(x1)
        high = torch.abs(x1 - F.interpolate(x2, size = x.size()[-2:], mode='bilinear', align_corners=True))
        return high


class ModulatedDeformConvWithOff(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int=3, 
                 stride: Optional[int]=1, 
                 padding: Optional[int]=1,
                 dilation: Optional[int]=1, 
                 deformable_groups: Optional[int]=1):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

        self.init_offset()

    def init_offset(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, input: Tensor):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output


class InvertedResidualSE(nn.Module):
    def __init__(
        self,
        inp_chan: int,
        expand_ratio: Optional[int] = 2,
        ffn_ratio: Optional[int] = 2,
        bias: Optional[bool] = False,
        drop_out_rate: Optional[float] = 0.,
        # skip_connection: Optional[bool] = True,
    ):
        super(InvertedResidualSE, self).__init__()
        dw_chan = inp_chan * expand_ratio
        ffn_chan = inp_chan * ffn_ratio

        self.norm1 = LayerNorm(inp_chan)
        self.norm2 = LayerNorm(inp_chan)
        
        self.pw1 = nn.Conv2d(inp_chan, dw_chan, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(dw_chan, dw_chan, kernel_size=3, padding=1, groups=dw_chan, bias=bias)

        self.sca = SimpleSeModule(dw_chan)

        self.pw2 = nn.Conv2d(dw_chan, inp_chan, kernel_size=1, bias=bias)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()

        self.ffn_conv1 = nn.Conv2d(inp_chan, ffn_chan, kernel_size=1, bias=bias)

        self.ffn_conv2 = nn.Conv2d(ffn_chan, inp_chan, kernel_size=1, bias=bias)

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, inp_chan, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, inp_chan, 1, 1)), requires_grad=True)

        self.inp_chan = inp_chan
        self.ffn_chan = ffn_chan
        self.dw_chan = dw_chan

    def forward(self, inp: Tensor) -> Tensor:
        # LayerNorm
        x = self.norm1(inp)
        # Depthwise Separable Convolutions
        x = self.pw1(x)
        x = self.dw(x)
        x = F.gelu(x)
        # Squeeze Channel Attention
        x = self.sca(x)
        x = self.pw2(x)

        x = self.dropout1(x)

        y = x * self.gamma + inp

        # ffn
        x = self.norm2(y)
        x = self.ffn_conv1(x)
        x = F.gelu(x)
        x = self.ffn_conv2(x)

        x = self.dropout2(x)

        oup = x * self.beta + y

        return oup
    

class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_w: int = 2,
        patch_h: int = 2,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv = nn.Conv2d(embed_dim, 1 + (2 * embed_dim), kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(1 + (2 * embed_dim), 1 + (2 * embed_dim), kernel_size=3, stride=1, padding=1, groups=1 + (2 * embed_dim), bias=bias)
        self.project_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)

        self.embed_dim = embed_dim

        self.patch_w = patch_w
        self.patch_h = patch_h

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)
    
    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        _, _, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:

        _, _, H, W = x.shape

        x = self.resize_input_if_needed(x)
        # [B, C, P, N] --> [B, h + 2d, P, N]
        patches, output_size = self.unfolding_pytorch(x)

        qkv = self.qkv_dwconv(self.qkv(patches))

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        # context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)

        out = self.project_out(out)

        out = self.folding_pytorch(out, output_size)

        return out[:, :, :H, :W]

    def forward(
        self, x: Tensor, *args, **kwargs
    ) -> Tensor:
        return self._forward_self_attn(x, *args, **kwargs)
    
    @staticmethod
    def visualize_context_scores(context_scores: Tensor):
        # context_scores_copy = context_scores.clone()
        # [B, 1, P, N]
        batch_size, channels, num_pixels, num_patches = context_scores.shape

        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
            channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels**0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        context_scores = context_scores[:, :, :, :patch_h * patch_w]
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.view(1, num_pixels, patch_h, patch_w)
        # Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        # [1, 1, H, W] --> [H, W]
        context_map = context_map.squeeze()

        # For ease of visualization, we do min-max normalization
        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)

        try:
            import os
            from glob import glob

            import cv2

            # convert from float to byte
            context_map = (context_map * 255).byte().cpu().numpy()
            context_map = cv2.resize(
                context_map, (80, 80), interpolation=cv2.INTER_NEAREST
            )

            colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
            # Lazy way to dump feature maps in attn_res folder. Make sure that directory is empty and copy
            # context maps before running on different image. Otherwise, attention maps will be overridden.
            res_dir_name = "attn_res"
            if not os.path.isdir(res_dir_name):
                os.makedirs(res_dir_name)
            f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

            files_cmap = glob(
                "{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
            )
            idx = len(files_cmap)
            f_name += str(idx)

            cv2.imwrite("{}.png".format(f_name), colored_context_map)
            return colored_context_map
        except ModuleNotFoundError as mnfe:
            print("Please install OpenCV to visualize context maps")
            return context_map


class FeedForward(nn.Module):
    def __init__(self, 
                 dim: int, 
                 ffn_expansion_factor: int, 
                 bias: bool):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 
                                kernel_size=3, stride=1, padding=1, 
                                groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # x = x1 * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 patch_w: int, 
                 patch_h: int, 
                 ffn_expansion_factor: int, 
                 bias: int):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, patch_h=patch_h, patch_w=patch_w, bias=bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@ARCH_REGISTRY.register()
class Reformer(nn.Module):
    def __init__(self, 
        inp_ch: Optional[int] = 3, 
        width: Optional[int] = 32,
        patch_w: Optional[int] = 2,
        patch_h: Optional[int] = 2,
        middle_blk_num: Optional[int]=1,
        middle_use_attn: Optional[bool] = 0,
        enc_blk_nums: Optional[List[int]] = [1, 1, 1, 28],
        enc_use_attns: Optional[List[bool]] = [0, 0, 0, 0],
        dec_blk_nums: Optional[List[int]] = [1, 1, 1, 1],
        dec_use_attns: Optional[List[bool]] = [0, 0, 0, 0],
        dw_expand: Optional[int] = 2,
        ffn_expand: Optional[int] = 2,
        bias: Optional[bool] = False, 
    ):
        super(Reformer, self).__init__()

        # self.intro = nn.Conv2d(in_channels=inp_ch, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                        bias=bias)

        self.intro = ModulatedDeformConvWithOff(in_channels=inp_ch, out_channels=width, kernel_size=3, padding=1)

        self.ending = nn.Conv2d(in_channels=width, out_channels=inp_ch + 1, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)

        # self.ending = ModulatedDeformConvWithOff(in_channels=width, out_channels=inp_ch, kernel_size=3, padding=1, stride=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.laters = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.reduce_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # self.dcn1 = ModulatedDeformConvWithOff(in_channels=width, out_channels=width, kernel_size=3, padding=1)
        # self.dcn2 = ModulatedDeformConvWithOff(in_channels=width, out_channels=width, kernel_size=3, padding=1)
        
        chan = width
        for num, enc_use_attn in zip(enc_blk_nums, enc_use_attns):
            self.downs.append(
                # DownSample(chan, bias=bias)
                nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2, bias=bias)
            )
            if enc_use_attn:
                self.encoders.append(nn.Sequential(
                    *[TransformerBlock(chan, 
                                       patch_h=patch_h, 
                                       patch_w=patch_w,
                                       ffn_expansion_factor=ffn_expand,
                                       bias=bias) for _ in range(num)]
                    )
                )

                # self.laters.append(nn.Identity())
                self.laters.append(HFBlock(chan, bias))

            else:
                self.encoders.append(
                    nn.Sequential(
                        *[InvertedResidualSE(chan, 
                                             expand_ratio=dw_expand, 
                                             ffn_ratio=ffn_expand) for _ in range(num)]
                    )
                )

                # self.laters.append(HFBlock())
                self.laters.append(nn.Conv2d(chan, chan, kernel_size=1, bias=bias))
            chan = chan * 2

        if middle_use_attn:
            self.middle_blks = nn.Sequential(
                        *[TransformerBlock(chan, 
                                        patch_h=patch_h, 
                                        patch_w=patch_w,
                                        ffn_expansion_factor=ffn_expand,
                                        bias=bias) for _ in range(middle_blk_num)]
                        )
            
        else:
            self.middle_blks = nn.Sequential(
                        *[InvertedResidualSE(chan, 
                                             expand_ratio=dw_expand, 
                                             ffn_ratio=ffn_expand) for _ in range(middle_blk_num)]
                        )


        for num, dec_use_attn in zip(dec_blk_nums, dec_use_attns):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=bias),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            self.reduce_blks.append(nn.Conv2d(2 * chan, chan, kernel_size=1, bias=bias))

            if dec_use_attn:
                self.decoders.append(nn.Sequential(
                    *[TransformerBlock(chan, 
                                       patch_h=patch_h, 
                                       patch_w=patch_w,
                                       ffn_expansion_factor=ffn_expand,
                                       bias=bias) for _ in range(num)]
                    )
                )

            else:
                self.decoders.append(
                    nn.Sequential(
                        *[InvertedResidualSE(chan, expand_ratio=dw_expand, ffn_ratio=ffn_expand) for _ in range(num)]
                    )
                )
        
        # self.dcn1 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)
        # self.dcn2 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward_features(self, x): #shape: 1 x 3 x 256 x 256    
        x = self.intro(x)

        encs = []

        for encoder, later, down in zip(self.encoders, self.laters, self.downs):
            x = encoder(x)
            encs.append(later(x))
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, reduce_blk, enc_skip in zip(self.decoders, self.ups, self.reduce_blks, encs[::-1]):
            x = up(x)
            x = reduce_blk(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)

        # x = self.dcn2(self.dcn1(x))
        # x = self.dcn1(x)
        
        x = self.ending(x)

        return x
    
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.check_image_size(x)

        feat = self.forward_features(x)

        K, N = torch.split(feat, (3, 1), dim=1)

        x = K * x + N + x

        return x[:, :, :H, :W]
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == "__main__":

    # inp_chan = 256
    # transformer_chan = 256
    # ffn_dim = transformer_chan * 2
    # n_attn_blocks = 4
    # patch_h = 2
    # patch_w = 2
    # conv_ksize = 3

    # net = MobileViTBlockv2(in_channels=inp_chan, transformer_dim=transformer_chan, 
    #                        ffn_dim=ffn_dim, n_attn_blocks=n_attn_blocks, attn_dropout=0.,
    #                        dropout=0., ffn_dropout=0., patch_h=patch_h, patch_w=patch_w, conv_ksize=conv_ksize)
    
    # net = MobileViTBlock(in_channels=inp_chan, transformer_dim=transformer_chan,
    #                      ffn_dim=ffn_dim, n_transformer_blocks=n_attn_blocks, head_dim=4, 
    #                      attn_dropout=0., dropout=0., ffn_dropout=0.,
    #                      patch_h=patch_h, patch_w=patch_w, conv_ksize=conv_ksize, no_fusion=False)

    net = Reformer(inp_ch=3,
                   width=16, 
                   patch_h=2,
                   patch_w=2,
                   middle_blk_num=8,
                   middle_use_attn=True,
                   enc_blk_nums=[1, 1, 1, 8],
                   enc_use_attns=[0, 0, 0, True],
                   dec_blk_nums=[1, 1, 1, 1],
                   dec_use_attns=[0, 0, 0, 0],
                   dw_expand=2,
                   ffn_expand=2,
                   bias=False)

    # inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # print(macs, params)

    # input = torch.randn((1, 3, 256, 256))

    # import thop

    # flops, params = thop.profile(net, inputs=(input, ))

    # # print (flops, params)
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    inp = torch.randn((1, 3, 256, 256))

    oup = net(inp)

    print (oup.shape)

