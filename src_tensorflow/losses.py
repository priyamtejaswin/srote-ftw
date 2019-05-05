#!/usr/bin/env python
"""
created at: 2019-05-02 14:01
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Training job.
"""


import numpy as np
import sys
import os
import tensorflow as tf

from model import ENHANCE
from utils import load_fnames
from utils import build_dataset


def combined_loss(pred_y, true_y):
    """
    Perceptual loss + Huber loss

    :param pred_y: Super-resolved image.
    :param true_y: High-res image
    :return: training loss.
    """
    pass


def huber_loss(flows, epsilon=0.01):
    """
    Huber loss. Eq 6.
    :param flows: Flow vectors (last dim==2; x & y)
    :param epsilon: Constant.
    :return:
    """
    del_x = flows[:, 1:] - flows[:, :-1]
    del_y = flows[:, :, 1:] - flows[:, :, :-1]

    del_x = tf.square(del_x)
    del_y = tf.square(del_y)

    batch_loss_sum = tf.sqrt(epsilon + tf.reduce_sum(del_x, axis=[1, 2, 3]) + tf.reduce_sum(del_y, axis=[1, 2, 3]))
    avg_batch_loss = tf.reduce_mean(batch_loss_sum)
    return avg_batch_loss


def main():
    frames_dir = sys.argv[1]
    if not os.path.isdir(frames_dir):
        raise OSError('Input path is not a directory or does not exist!')

    frames_list = load_fnames(frames_dir)
    dataset = build_dataset(frames_list[:3])

    model = ENHANCE()


if __name__ == '__main__':
    main()