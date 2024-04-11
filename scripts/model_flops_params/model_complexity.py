# -*- encoding: utf-8 -*-
'''
@File    :   model_complexity.py
@Time    :   2024/01/17 19:36:07
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib

import torch
import ptflops
import thop
from os import path as osp
import logging
import argparse
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.utils.options import yaml_load

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    return opt


def main(root_path):

    opt = parse_options()

    net = build_network(opt=opt["network_g"])
    log_file = osp.join(root_path, f"{net.__class__.__name__}_complexity.log")

    logger = get_root_logger(logger_name="model_complexity", 
                             log_level=logging.INFO, log_file=log_file)
    logger.info(f"Network {net.__class__.__name__} is created.")

    inp_shape = (3, 256, 256)

    macs, params = ptflops.get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    logger.info("Used ptflops, MACs = {0:s}, params = {1:s}".format(macs, params))

    input = torch.randn((1, 3, 256, 256))

    flops, params = thop.profile(net, inputs=(input, ), verbose=False)

    flops /= 1e9
    params /= 1e6

    logger.info("Used thop, FLOPs = {0:.3f}G, params = {1:.3f}M".format(flops, params))

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    main(root_path=root_path)
