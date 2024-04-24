# -*- encoding: utf-8 -*-
'''
@File    :   infer_simple.py
@Time    :   2024/01/09 19:38:53
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib

import contextlib
import sys
import torch
import argparse
import os
import cv2 as cv
import numpy as np
import torch.nn as nn
import torchvision
from collections import OrderedDict
import albumentations as albu
from basicsr.utils import tensor2img, imwrite, img2tensor
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.archs.reformer_arch import Reformer
from basicsr.archs.NAFNet_arch import NAFNetLocal
from basicsr.archs.restormer_arch import Restormer
from basicsr.archs.fftformer_arch import fftformer
from basicsr.archs.hinet_arch import HINet
from basicsr.archs.uformer_arch import Uformer
from basicsr.archs.MiMO_unet_arch import MIMOUNet, MIMOUNetPlus
from basicsr.archs.deblurganv2_arch import FPNInception
from basicsr.archs.MRN_arch import MPRNet
from basicsr.archs.stripformer_arch import Stripformer
from basicsr.archs.minet_arch import MiNet
import time
from tqdm import tqdm
from timeit import default_timer as timer
from torch.utils.benchmark import Timer


class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield
    sys.stdout = old_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', "-m", type=str, default='experiments/MiNet-v3-GoPro/models/net_g_latest.pth')
    parser.add_argument('--input_path', "-i", type=str, default='datasets/CAL/train/hazy',
                        help='input test image folder')
    parser.add_argument('--output_path', "-o", type=str, default='images/', help='save image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs_file_path = args.input_path
    output_path = args.output_path

    model = MiNet()

    # model = Reformer(inp_ch=3,
    #                     width=16, 
    #                     patch_h=2,
    #                     patch_w=2,
    #                     middle_blk_num=8,
    #                     middle_use_attn=True,
    #                     enc_blk_nums=[1, 1, 1, 8],
    #                     enc_use_attns=[0, 0, 0, True],
    #                     dec_blk_nums=[1, 1, 1, 1],
    #                     dec_use_attns=[0, 0, 0, 0],
    #                     dw_expand=2,
    #                     ffn_expand=2,
    #                     bias=False)

    # model = NAFNetLocal(width=64, 
    #                     enc_blk_nums = [1, 1, 1, 28],
    #                     middle_blk_num = 1,
    #                     dec_blk_nums = [1, 1, 1, 1])

    # model = Restormer(inp_channels=3, 
    #                   out_channels=3, 
    #                   dim=48,
    #                   num_blocks=[4, 6, 6, 8],
    #                   num_refinement_blocks=4,
    #                   heads=[1, 2, 4, 8],
    #                   ffn_expansion_factor=2.66,
    #                   bias=False,
    #                   LayerNorm_type="WithBias",
    #                   dual_pixel_task=False)

    # model = fftformer(inp_channels=3, 
    #                   out_channels=3, 
    #                   dim=48, 
    #                   num_blocks=[6, 6, 12],
    #                   num_refinement_blocks=4,
    #                   ffn_expansion_factor=3,
    #                   bias=False)

    # model = HINet(wf=64, hin_position_left=3, hin_position_right=4)

    # model = Uformer(img_size=128, embed_dim=32, win_size=8, token_projection="linear", token_mlp="leff",
    #                 depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)

    # model = MIMOUNet()
    # model = MIMOUNetPlus()

    # model = FPNInception(norm_layer=torch.nn.InstanceNorm2d)
    # model = MPRNet()
    # model = nn.DataParallel(Stripformer())
    # model = nn.DataParallel(model)

    # model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=True) # MPRNet
    # model.load_state_dict(torch.load(args.model_path)['model'], strict=True) # MIMOUNet
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)  # Reformer NAFNet HINet MINet
    # model.load_state_dict(torch.load(args.model_path), strict=True) # Stripformer FFTformer
    model.to(device=device)
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    ## 2. read images dir
    print(f"Deleting files with 'deblur' and 'dehazed' suffix in {imgs_file_path}")

    try:
        files_to_delete = [f for f in os.listdir(imgs_file_path) if f.endswith("deblur.png") or f.endswith("dehazed.png")]
        for file_name in files_to_delete:
            file_path = os.path.join(imgs_file_path, file_name)
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

    images_name: str = os.listdir(imgs_file_path)
    # images_name = sorted(os.listdir(imgs_file_path), key=lambda name: int(''.join(filter(str.isdigit, name))))

    print("total %d images" % len(images_name))
    time.sleep(1)
    print("Working!.........")

    pbar = tqdm(images_name, total=len(images_name), unit="image")

    with torch.no_grad():
        for img_name in pbar:
            pbar.set_description(f'infer {img_name}')
            img = cv.imread(os.path.join(imgs_file_path, img_name), cv.IMREAD_COLOR).astype(np.float32) / 255.
            # img = np.random.normal(0, 1, size=(256, 256, 3)).astype(np.float32)
            # img = img2tensor(img, bgr2rgb=True, float32=True)
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0)
            img = img.to(device)

            ##3. inference 
            starter.record()
            output = model(img)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            sr_img = tensor2img(output)

            # imwrite(sr_img, os.path.join(output_path, img_name[:-4] + "_deblur.png"))
            # imwrite(sr_img, os.path.join(output_path, img_name))
            torchvision.utils.save_image(output, os.path.join(output_path, img_name))
            with redirect_stdout():
                print(f'inference {img_name} .. finished,'
                      f'elapsed {(starter.elapsed_time(ender)):.5f} millisecond.'
                      f'saved to {output_path}')

    print("Done!")


if __name__ == '__main__':
    main()
