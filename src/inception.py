import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
from symbol import *


if __name__ == '__main__':


    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'mlp',
        #num_layers     = 110,

        num_classes    = 8,
        num_examples  = 3021,
        image_shape    = '3,224,224',
        pad_size       = 4,
        # train
        batch_size     = 20,
        num_epochs     = 300,
        lr             = .05,
        lr_step_epochs = '20,25',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbol.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)