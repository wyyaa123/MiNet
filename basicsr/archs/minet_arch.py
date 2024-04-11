import math
import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
from torchvision.ops import DeformConv2d
from basicsr.archs.arch_util import LayerNorm
from basicsr.utils.registry import ARCH_REGISTRY
from typing import Optional, Tuple, Union, Dict, List, Sequence


class SimpleCaModule(nn.Module):
    def __init__(self, inp_chan: int):
        super(SimpleCaModule, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_chan, inp_chan, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        return x * self.ca(x)
    

class SimplePaModule(nn.Module):
    def __init__(self, inp_chan: int) -> None:
        super(SimplePaModule, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(inp_chan, inp_chan, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        return x * self.pa(x)


class HFBlock(nn.Module):
    def __init__(self, 
                 inp_chan: int, 
                 bias: bool = True):
        super(HFBlock, self).__init__()
        self.conv = nn.Conv2d(inp_chan, inp_chan, kernel_size=1, bias=bias)
        self.down = nn.AvgPool2d(2)
    def forward(self, x: Tensor):
        x1 = self.conv(x)
        x2 = self.down(x1)
        high = torch.abs(x1 - F.interpolate(x2, size = x.size()[-2:], mode='bilinear', align_corners=True))
        return high


class ModulatedDeformConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int=3, 
                 stride: Optional[int]=1, 
                 padding: Optional[int]=1,
                 dilation: Optional[int]=1, 
                 deformable_groups: Optional[int]=1):
        super(ModulatedDeformConv, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=deformable_groups,
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
    

class BasicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size = 3,
                 padding = 2,
                 dilation = 2) -> None:
        super(BasicConv, self).__init__()

        self.norm1 = LayerNorm(in_channels)
        self.norm2 = LayerNorm(in_channels)

        self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(in_channels, in_channels, 
                            kernel_size=kernel_size, padding=padding, 
                            dilation=dilation, groups=in_channels)

        self.pa = SimplePaModule(in_channels)
        self.ca = SimpleCaModule(in_channels)

        self.proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, input: Tensor):

        res = input
        x = self.norm1(input)
        x = self.dw(self.pw(x))
        x = self.act(x)
        x = torch.cat([self.ca(x), self.pa(x)], dim=1)
        x = self.proj(x)

        x = x * self.gamma + res

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x * self.beta + res

        return x
    

class LinearSelfAttention(nn.Module):
    """
    modifed from https://github.com/apple/ml-cvnets.git

    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """
    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = nn.Conv2d(embed_dim, 1 + (2 * embed_dim), kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv_proj = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv_proj, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)

        return out

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
                 embed_dim,
                 ffn_latent_dim,
                 ffn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, 
                      out_channels=ffn_latent_dim, 
                      kernel_size=1, 
                      bias=True),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(in_channels=ffn_latent_dim, 
                      out_channels=embed_dim, 
                      kernel_size=1, 
                      bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        return self.ffn(x)


class LinearAttnFFN(nn.Module):
    """
    modifed from https://github.com/apple/ml-cvnets.git

    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            LayerNorm(embed_dim),
            FeedForward(embed_dim=embed_dim, 
                        ffn_latent_dim=ffn_latent_dim,
                        ffn_dropout=ffn_dropout,
                        dropout=dropout)
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        
        # self-attention
        x = x + self.pre_norm_attn(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockv3(nn.Module):
    """
    This class defines the `MobileViTv3 <>`_ block

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 4,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 2,
        patch_w: Optional[int] = 2,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                dilation=dilation,
                groups=in_channels,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.GELU(),
        )

        conv_1x1_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        super(MobileViTBlockv3, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_proj = nn.Sequential(
            nn.Conv2d(
                in_channels= 2 * cnn_out_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1
                ),
            nn.BatchNorm2d(num_features=in_channels)
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            LayerNorm(normalized_shape=d_model)
        )

        return nn.Sequential(*global_rep), d_model


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
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        _, _, H, W = x.shape
        x = self.resize_input_if_needed(x)

        fm_conv = self.local_rep(x)

        # convert feature map to patches
        patches, output_size = self.unfolding_pytorch(fm_conv)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        fm = self.conv_proj(torch.cat([fm, fm_conv], dim=1))

        fm = fm + x

        return fm[:, :, :H, :W]


    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.forward_spatial(x)
    

class UpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, up_ratio = 2) -> None:
        super(UpsampleModule, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * up_ratio ** 2, kernel_size=1),
            nn.PixelShuffle(up_ratio)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


@ARCH_REGISTRY.register()
class MiNet(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[2, 4, 4, 2, 2]) -> None:
        super(MiNet, self).__init__()

        self.intro = nn.Conv2d(in_chans, embed_dims[0], kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(*[BasicConv(embed_dims[0]) for _ in range(depths[0])])

        self.down1 = nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=2, stride=2)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = MobileViTBlockv3(embed_dims[1], embed_dims[1], n_attn_blocks=depths[1])

        self.down2 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=2, stride=2)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = MobileViTBlockv3(embed_dims[2], embed_dims[2], n_attn_blocks=depths[2])

        self.up1 = UpsampleModule(embed_dims[2], embed_dims[3], up_ratio=2)

        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = nn.Sequential(*[BasicConv(embed_dims[3]) for _ in range(depths[3])])

        self.up2 = UpsampleModule(embed_dims[3], embed_dims[4], up_ratio=2)

        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = nn.Sequential(*[BasicConv(embed_dims[4]) for _ in range(depths[4])])

        self.end = nn.Conv2d(embed_dims[4], out_chans, kernel_size=3, padding=1)

        self.padder_size = len(depths) // 2

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward_features(self, x):
        x = self.intro(x)
        x = self.layer1(x)
        skip1 = x

        x = self.down1(x)
        x = self.layer2(x) + x
        skip2 = x

        x = self.down2(x)
        x = self.layer3(x) + x
        x = self.up1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.up2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.end(x)

        return x
    
    def forward(self, x: Tensor):
        _, _, H, W = x.shape
        x = self.check_image_size(x)

        feat = self.forward_features(x)
        # 2022/11/26
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


if __name__ == "__main__":

    net = MiNet()

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

    net = net.cuda()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    inp = torch.randn((1, 3, 548, 412), device='cuda:0')

    starter.record()
    oup = net(inp)
    ender.record()

    torch.cuda.synchronize()

    print (f'elapsed {(starter.elapsed_time(ender)):.5f} millisecond.')

