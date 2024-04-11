import torch
from basicsr.archs.reformer_arch import Reformer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', "-m", type=str, 
                    default='experiments/pretrained_models/HINet-GoPro.pth')
args = parser.parse_args()

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
# model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

model.eval()

input = torch.randn((1, 3, 420, 270))
output = model(input)

torch.onnx.export(
    model, (input,), "test.onnx", verbose=True, opset_version=11
)
