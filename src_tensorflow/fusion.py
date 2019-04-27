#!/usr/bin/env python
"""
created at: 27/04/19 7:07 PM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Classes for different Spatio-Temporal Models.
<paper>
Refer to Fig.4 for comparision of architectures.
"""


import tensorflow as tf
keras = tf.keras
from keras.layers import TimeDistributed
from keras.layers import Conv2D
from keras.activations import  relu


def EarlyFusion(timeframes, iheight, iwidth, channels=3,
                nfilters=None, ksize=3, activation=relu):
    """
    A wrapper over TimeDistributed layer for Early Fusion.
    Returns a TimeDistributed Conv2d layer you can add to the model.

    :param timeframes: Number of timesteps/frames; the first index AFTER batch.
    :param iheight: Height of frame.
    :param iwidth: Width of frame.
    :param channels: Image channels.
    :param nfilters: Filters for Conv2d; will be imputed if None.
    :param ksize: Kernel size for Conv2d.
    :param activation: Activation for Conv2d.
    :return: A TimeDistributed Conv2d layer you can add to the model.
    """
    if nfilters is None:
        nfilters = 24 // timeframes

    return TimeDistributed(
        Conv2D(nfilters, ksize, activation=activation),
        input_shape=(timeframes, iheight, iwidth, channels)
    )


if __name__ == '__main__':
    print EarlyFusion(
        timeframes=3, iheight=32, iwidth=32, channels=3
    )