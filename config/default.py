# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()
_C.nn = 10000 # TODO: number of training data
_C.num_exp = 5 # TODO: number of repeats
_C.NUM_EPOCHES = 100
_C.batch_size = 1000
_C.p = 50
_C.d = 10
_C.m = 50
_C.lr = 0.0001
_C.has_online_ReLU = True
_C.has_target_ReLU = True
_C.has_target_predictor = False
_C.alternative_optimization = True
_C.rm_predictor = True
_C.sigma0 = None
_C.threshold = 0 #Used in sparse coding model by Arora et.al.
_C.z_dim = None  # Only dimensions 0~z_dim can be non-zero in z
_C.one_hot_latent = False  # If true, there is exactly one +/-1 in z, all others are 0
_C.batch_norm = None  # None, 'encoder_pre_activation', or 'encoder_out'

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.lr:
        cfg.lr = args.lr 
    
    if args.batch_size:
        cfg.batch_size = args.batch_size

    print(cfg)
    
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

