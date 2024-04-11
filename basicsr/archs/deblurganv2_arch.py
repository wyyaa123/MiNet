import torch
import torch.nn as nn
import numpy as np
from pretrainedmodels import inceptionresnetv2
from torchsummary import summary
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.block0(x), inplace=True)
        x = F.relu(self.block1(x), inplace=True)
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_in, num_out, norm_layer):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
                                 norm_layer(num_out),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


@ARCH_REGISTRY.register()
class FPNInception(nn.Module):

    def __init__(self, norm_layer=nn.InstanceNorm2d, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

        self.padder_size = 2 ** 5


    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.check_image_size(x)

        map0, map1, map2, map3, map4 = self.fpn(x)

        # map4 = F.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map4 = F.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        # map3 = F.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map3 = F.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        # map2 = F.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map2 = F.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        # map1 = F.upsample(self.head1(map1), scale_factor=1, mode="nearest")
        map1 = F.interpolate(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        # smoothed = F.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        # smoothed = F.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        res = torch.clamp(res, min = -1,max = 1)

        return res[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w= x.shape
        min_height = (h // self.padder_size + 1) * self.padder_size
        min_width = (w // self.padder_size + 1) * self.padder_size
        x = F.pad(x, (0, min_width - w, 0, min_height - h)) ### modified
        return x


class FPN(nn.Module):
    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        ) # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0) # 256

        enc2 = self.enc2(enc1) # 512

        enc3 = self.enc3(enc2) # 1024

        enc4 = self.enc4(enc3) # 2048

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        # map3 = self.td1(lateral3 + F.upsample(map4, scale_factor=2, mode="nearest"))
        map3 = self.td1(lateral3 + F.interpolate(map4, scale_factor=2, mode="nearest"))
        # map2 = self.td2(F.pad(lateral2, pad, "reflect") + F.upsample(map3, scale_factor=2, mode="nearest"))
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + F.interpolate(map3, scale_factor=2, mode="nearest"))
        # map1 = self.td3(lateral1 + F.upsample(map2, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + F.interpolate(map2, scale_factor=2, mode="nearest"))
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4
    

if __name__ == "__main__":

    net = FPNInception(norm_layer=nn.InstanceNorm2d)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

    input = torch.randn((1, 3, 256, 256))

    import thop

    flops, params = thop.profile(net, inputs=(input, ))

    # print (flops, params)
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
