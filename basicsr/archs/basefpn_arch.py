import torch
import torch.nn as nn
# from basicsr.archs.arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
from mmcv.ops import ModulatedDeformConv2d
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# mobileNet_v3
class BaseBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(BaseBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class BaseFPN(nn.Module):
    def __init__(self, img_channel, width=16):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, 
                               kernel_size=3, padding=1, bias=True)
        

        self.bneck = nn.Sequential(
            BaseBlock(kernel_size=3, in_size=width, expand_size=2 * width, out_size=2 * width, act=nn.ReLU, se=True, stride=2),

            BaseBlock(kernel_size=3, in_size=2 * width, expand_size=4 * width, out_size=2 * width, act=nn.ReLU, se=True, stride=1),
            ######################################################################################################
            BaseBlock(kernel_size=3, in_size=2 * width, expand_size=4 * width, out_size=4 * width, act=nn.ReLU, se=True, stride=2),
            BaseBlock(kernel_size=3, in_size=4 * width, expand_size=8 * width, out_size=4 * width, act=nn.ReLU, se=True, stride=1),

            BaseBlock(kernel_size=3, in_size=4 * width, expand_size=8 * width, out_size=4 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=4 * width, expand_size=8 * width, out_size=4 * width, act=nn.ReLU, se=True, stride=1),
            ######################################################################################################
            BaseBlock(kernel_size=3, in_size=4 * width, expand_size=8 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=2),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=8 * width, act=nn.ReLU, se=True, stride=1),
            ######################################################################################################
            BaseBlock(kernel_size=3, in_size=8 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=2),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),

            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
            BaseBlock(kernel_size=3, in_size=16 * width, expand_size=16 * width, out_size=16 * width, act=nn.ReLU, se=True, stride=1),
        )

        self.encoder0 = nn.Sequential(self.bneck[0:2]) # inp: 16, oup: 32, shape: down
        self.encoder1 = nn.Sequential(self.bneck[2:6]) # inp: 32, oup: 64, shape: down
        self.encoder2 = nn.Sequential(self.bneck[6:14]) # inp: 64, oup: 128, shape: down
        self.encoder3 = nn.Sequential(self.bneck[14:])  # inp: 128, oup: 256, shape: down

        self.lateral3 = nn.Sequential(nn.Conv2d(16 * width, 32 * width, kernel_size=1, bias=False),
                                      nn.PixelShuffle(2)) # inp: 256, oup: 128, shape: 1/8
        
        self.lateral2 = nn.Sequential(nn.Conv2d(8 * width, 16 * width, kernel_size=1, bias=False),
                                      nn.PixelShuffle(2)) # inp: 128, oup: 64, shape: 1/4
        
        self.lateral1 = nn.Sequential(nn.Conv2d(4 * width, 8 * width, kernel_size=1, bias=False),
                                      nn.PixelShuffle(2)) # inp: 64, oup: 32, shape: 1/2
        
        self.lateral0 = nn.Sequential(nn.Conv2d(2 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.PixelShuffle(2)) # inp: 32, oup: 16, shape: -
        

        self.td0 = nn.Sequential(nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(2 * width),
                                 nn.ReLU(inplace=True)) # inp: 32, oup: 32

        self.td1 = nn.Sequential(nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(4 * width),
                                 nn.ReLU(inplace=True)) # inp: 64, oup: 64
        
        self.td2 = nn.Sequential(nn.Conv2d(8 * width, 8 * width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(8 * width),
                                 nn.ReLU(inplace=True)) # inp: 128, oup: 128
        
        self.td3 = nn.Sequential(nn.Conv2d(16 * width, 16 * width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(16 * width),
                                 nn.ReLU(inplace=True)) # inp: 256, oup: 256
        
        self.padder_size = 2 ** 4

    def forward(self, inp):

        inp = self.intro(inp) # inp: 3, oup: 16, shape: -

        # Bottom-up pathway, from MobileNet_v3
        enc0 = self.encoder0(inp) # inp: 16, oup: 32, shape: 1/2

        enc1 = self.encoder1(enc0) # inp: 32, oup: 64, shape: 1/4

        enc2 = self.encoder2(enc1) # inp: 64, oup: 128, shape: 1/8

        enc3 = self.encoder3(enc2) # inp: 128, oup: 256, shape: 1/16

        # Top-down pathway
        map3 = self.td3(enc3) # inp: 256, oup: 256, shape: 1/16
        map2 = self.td2(enc2 + self.lateral3(map3)) # inp: 128, oup: 128, shape: 1/8
        map1 = self.td1(enc1 + self.lateral2(map2)) # inp: 64, oup: 64, shape: 1/4
        map0 = self.td0(enc0 + self.lateral1(map1)) # inp: 32, oup: 32, shape: 1/2
        return map0, map1, map2, map3 
    

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        if num_in == num_out:
            self.block0 = nn.Sequential(nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False),
                                        nn.ReLU(inplace=True))
        else:
            self.block0 = nn.Sequential(nn.Conv2d(num_in, num_mid, kernel_size=2, padding=1, bias=False, dilation=2),
                                        nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x
    

class ModulatedDeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
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

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output
    
# @ARCH_REGISTRY.register()
class BaseFPNNet(nn.Module):

    def __init__(self, inp_ch, output_ch=3, width=16):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/2, 1/4, 1/8, 1/16 and `width` filters for all feature maps.

        self.fpn = BaseFPN(inp_ch)

        # The segmentation heads on top of the FPN
        self.head0 = nn.Sequential(FPNHead(2 * width, 2 * width, 2 * width),
                                   nn.PixelShuffle(1)) # inp: 32, oup: 32, shape: 1/2
        self.head1 = nn.Sequential(FPNHead(4 * width, 4 * width, 4 * width),
                                   nn.PixelShuffle(2)) # inp: 64, oup: 16, shape: 1/2
        self.head2 = nn.Sequential(FPNHead(8 * width, 8 * width, 8 * width),
                                   nn.PixelShuffle(4)) # inp: 128, oup: 8, shape: 1/2
        self.head3 = nn.Sequential(FPNHead(16 * width, 32 * width, 32 * width),
                                   nn.PixelShuffle(8)) # inp: 256, oup: 8, shape: 1/2

        self.dcn1 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)
        self.dcn2 = ModulatedDeformConvWithOff(width * 4, width * 4, kernel_size=3, padding=1, stride=1, deformable_groups=4)

        self.ending = nn.Conv2d(in_channels=width, out_channels=output_ch, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.padder_size = 2 ** 4

    def forward(self, inp):

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        map0, map1, map2, map3 = self.fpn(inp) # 32 64 128 256

        map3 = self.head3(map3)
        map2 = self.head2(map2)
        map1 = self.head1(map1)
        map0 = self.head0(map0)

        oup = torch.cat([map3, map2, map1, map0], dim=1)

        oup = self.dcn1(oup)

        oup = self.dcn2(oup)

        oup = nn.PixelShuffle(2)(oup)

        oup = self.ending(oup) + inp

        return oup[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size #检查宽高是否为padder_size的倍数
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

if __name__ == "__main__":
    net = BaseBlock(kernel_size=3, in_size=16, 
                    expand_size=32, out_size=16,
                    act=nn.ReLU, se=True, stride=2)

    inp_shape = (16, 224, 224)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

    # inp = torch.randn((1, 3, 256, 256))

    # # net = nn.PixelShuffle(8)

    # oup = net(inp)

    # print()

