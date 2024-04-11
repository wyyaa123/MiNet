import time
from tqdm import tqdm
import contextlib
import sys
import torch
import argparse
import os
import torch
from torch import nn
import cv2 as cv
import numpy as np
from basicsr.archs.reformer_arch import Reformer
from basicsr.utils.show_offset import show_dconv_offset

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

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', "-m", type=str, default='experiments/ReformerNet-v1-GoPro/models/net_g_latest.pth')
    parser.add_argument('--input_path', "-i", type=str, default='images/1.png', help='input test image path')
    parser.add_argument('--output_path', "-o", type=str, default='images/1.png', help='save image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = args.input_path
    output_path = args.output_path

    save_output = SaveOutput()

    hook_handles = []

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
    # model.to(device=device)

    for name, layer in model.named_modules():
        if "offset_mask_conv" in name and isinstance(layer, nn.Conv2d):
            print (name, layer)
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    img = cv.imread(img_path, cv.IMREAD_COLOR).astype(np.float32) / 255.
    un_img = np.copy(img)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0)
    # img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    print (len(save_output.outputs))

    o1, o2, mask = torch.chunk(save_output.outputs[0], 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = torch.sigmoid(mask)

    offset = offset.detach().cpu().numpy()

    un_img = cv.cvtColor(un_img, cv.COLOR_BGR2RGB)

    show_dconv_offset(un_img , [offset], dilation=1, pad=1, plot_level=1)


if __name__ == "__main__":
    main()
