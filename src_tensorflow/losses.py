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
from tensorflow.keras.applications import vgg16
from tensorflow.keras import Model


def perceptual_loss_wrapper(vgg_layer="block2_conv2"):
    vgg = vgg16.VGG16(include_top=False, weights="imagenet")
    perceptual_layer_model = Model(
        inputs=vgg.input, 
        outputs=vgg.get_layer(vgg_layer).output)
    perceptual_layer_model.trainable=False 
    perceptual_layer_model.compile(loss="mse", optimizer="adam")
    
    def perceptual_loss(y_true, y_pred):
        y_true_vgg = perceptual_layer_model(y_true)
        y_pred_vgg = perceptual_layer_model(y_pred)
        return tf.reduce_mean(tf.square(y_pred_vgg - y_true_vgg))
    
    return perceptual_loss


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


class CombinedLoss:
    def __init__(self, alpha=0.0, beta=0.0, gamma=1.0, lam=1.0):
        """
        alpha: multiplier for basic mse loss 
        beta: multiplier for spatial motion compensation loss (like in paper)
        gamma: multiplier for vgg16 perceptual loss 
        lam: multiplier for huber loss 
        """ 
        self._perceptual_loss_function = perceptual_loss_wrapper() 
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma 
        self.lam = lam 
    
    def mse(self, x, y):
        return tf.reduce_mean(tf.square(x-y))

    def all_loss(self, y_true, y_pred, flow1, flow2, comp1, comp2, frames):
        """
        y_true: the high-res ground truth image 
        y_pred: the high-res model predicted image 
        flow1: flow vector for i_t and i_t+1 
        flow2: flow vector for i_t and i_t-1 
        """
        # basic mse loss 
        basic_mse = self.mse(y_pred, y_true)

        # motion comp
        comp1_mse = self.beta * self.mse(comp1, frames[:, 0])
        comp2_mse = self.beta * self.mse(comp2, frames[:, 2])

        # huber loss
        huber1 = huber_loss(flow1)
        huber2 = huber_loss(flow2)

        # perceptual loss 
        perc_mse = self._perceptual_loss_function(y_true, y_pred)

        # combine the losses 
        full_loss = (self.alpha * basic_mse) + (self.beta * comp1_mse) \
                    + (self.lam * huber1) + (self.beta * comp2_mse) \
                    + (self.lam * huber2) + (self.gamma * perc_mse)
        
        return full_loss


if __name__ == '__main__':
    tf.enable_eager_execution()
    print 'Tf executing eagerly?', tf.executing_eagerly()

    tf.set_random_seed(1)

    fnames = load_fnames('../data/frames')
    dataset = build_dataset(fnames[:5])

    model = ENHANCE()
    optimizer = tf.train.AdamOptimizer()
    percept_loss = perceptual_loss_wrapper()
    combined_loss = CombinedLoss().all_loss

    for epoch in range(1):
        for ix, (x,y) in enumerate(dataset.take(10)):
            with tf.GradientTape() as tape:
                preds, (flow1, flow2), (comp1, comp2) = model(x)
                # loss = huber_loss(flow1) + huber_loss(flow2)
                # loss = loss + percept_loss(y_true=y, y_pred=preds)
                loss = combined_loss(y_true=y, y_pred=preds,
                                     flow1=flow1, flow2=flow2,
                                     comp1=comp1, comp2=comp2,
                                     frames=x)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            if ix%1 == 0:
                print 'Epoch: {}, Step: {}, Loss: {}'.format(epoch+1, ix+1, loss.numpy())

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