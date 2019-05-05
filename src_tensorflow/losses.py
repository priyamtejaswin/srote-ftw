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


def combined_loss(model_out, true_y):
    pred_y, (flow1, flow2) = model_out
    flow_loss = huber_loss(flow1) + huber_loss(flow2)
    return flow_loss


if __name__ == '__main__':
    tf.enable_eager_execution()
    print 'Tf executing eagerly?', tf.executing_eagerly()

    fnames = load_fnames('../data/frames')
    dataset = build_dataset(fnames[:5])

    model = ENHANCE()
    optimizer = tf.train.AdamOptimizer()

    for ix, (x,y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds, (flow1, flow2) = model(x)
            loss = huber_loss(flow1) + huber_loss(flow2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        print 'Step: {}, Loss: {}'.format(ix, loss.numpy())

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # @tf.py_function
    # def train_step(low_res, high_res):
    #     with tf.GradientTape() as tape:
    #         upped, (flow1, flow2) = model(low_res)
    #         loss = huber_loss(flow1) + huber_loss(flow2)
    #
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #
    # for epoch in range(5):
    #     for x,y in dataset:
    #         train_step(x, y)
    #
    #     template = 'Epoch {}, Loss: {}'
    #     print template.format(epoch+1, train_loss.result())