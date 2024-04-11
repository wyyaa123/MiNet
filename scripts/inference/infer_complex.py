# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import os
from tqdm import tqdm
import contextlib
import sys
from os import path as osp
from basicsr.utils import tensor2img, imwrite, FileClient, imfrombytes, img2tensor, padding, get_root_logger
from basicsr.models import build_model
import time
from basicsr.utils.options import parse_options

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

def main(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path=root_path, is_train=False)
    # opt['num_gpu'] = torch.cuda.device_count()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs_file_path = opt['img_path'].get('input_img_dir')
    output_file_path = opt['img_path'].get('output_img_dir')
    images_name = sorted(os.listdir(imgs_file_path), key=lambda name: int(''.join(filter(str.isdigit, name))))

    print ("total %d images" % len(images_name))
    time.sleep(1)
    print ("Working!.........")

    file_client = FileClient('disk') 

    ## 1. build model
    model = build_model(opt)

    pbar = tqdm(images_name, total=len(images_name), unit="image")
    for img_name in pbar:

        pbar.set_description(f'infer {img_name}')

        img_path = imgs_file_path + img_name
        # output_path = output_file_path + img_name
        output_path = output_file_path + img_name[:-4] + "_deblur.png"

        ## 2. read image
        img_bytes = file_client.get(img_path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        beg = time.time()
        img = img2tensor(img, bgr2rgb=True, float32=True)

        ## 2. run inference
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.nonpad_test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        # model.single_image_inference(img, output_path)
        end = time.time()

        imwrite(sr_img, output_path)

        with redirect_stdout():
                print(f'inference {img_name} .. finished, elapsed {(end - beg):.3f} seconds. saved to {output_path}')

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)

