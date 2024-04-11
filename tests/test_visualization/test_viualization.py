# -*- encoding: utf-8 -*-
'''
@File    :   test_viualization.py
@Time    :   2023/12/26 21:23:52
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import torch
import os
import time
import argparse
import cv2 as cv
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from basicsr.archs.reformer_arch import Reformer
from basicsr.utils.save_output import SaveOutput

def scaled(matrix: np.ndarray):
    # 找到矩阵的最小值和最大值
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    # 将矩阵的值缩放到 0 到 1 的范围
    scaled_matrix = (matrix - min_value) / (max_value - min_value)

    return scaled_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', "-m", type=str, default='experiments/ReformerNet-v1-GoPro/models/net_g_latest.pth')
    parser.add_argument('--input_path', "-i", type=str, default='images/1413394989805760512.png', help='input test image path')
    parser.add_argument('--output_path', "-o", type=str, default='images/1.png', help='save image path')
    args = parser.parse_args()

    img_path = args.input_path
    output_path = args.output_path

    hook_handles = []

    images = []

    save_output = SaveOutput()

    model = Reformer(inp_ch=3,
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
    
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    for name, layer in model.named_modules():
        if "laters.3.conv" in name and isinstance(layer, nn.Conv2d):
            print (name, layer)
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    img = cv.imread(img_path, cv.IMREAD_COLOR).astype(np.float32) / 255.
    un_img = np.copy(img)[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0)
    # img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    print (len(save_output.outputs))

    TL = save_output.outputs[-1]

    avg_map = F.avg_pool2d(TL, kernel_size=2, stride=2)

    TU = F.interpolate(avg_map, size=TL.size()[-2:], mode='bilinear', align_corners=True)

    high_map = TU - TL

    for fm in [TL, TU, high_map]:
        fm = torch.abs(fm)
        fm = fm.squeeze(0)
        gray_scale = torch.sum(fm, dim=0)
        gray_scale = gray_scale / fm.shape[0]
        gray_scale = gray_scale.data.cpu().numpy()
        images.append(gray_scale)

    # fig, axs = plt.subplots(2, 2, constrained_layout=True, squeeze=False)


    # plt.subplot(2, 2, 1)
    # plt.title('blur')
    plt.imshow(un_img)
    plt.axis('off')
    plt.savefig("un_img.png")

    # plt.subplot(2, 2, 2)
    # plt.title('M_in')
    plt.imshow(images[0])
    plt.axis('off')
    plt.savefig("m_in.png")

    # plt.subplot(2, 2, 3)
    # plt.title('M_up')
    plt.imshow(images[1])
    plt.axis('off')
    plt.savefig("m_up.png")

    # plt.subplot(2, 2, 4)
    # plt.title('M_High')
    plt.imshow(images[2])
    plt.axis('off')
    plt.savefig("m_high.png")

    # plt.tight_layout()

    plt.show()




