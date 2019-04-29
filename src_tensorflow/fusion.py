#!/usr/bin/env python
"""
created at: 27/04/19 7:07 PM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Classes for different Spatio-Temporal Models.
<paper>
Refer to Fig.4 for comparision of architectures.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import linear
from tensorflow.keras.layers import Layer


def early_fusion(timeframes, iheight, iwidth, channels=3,
                nfilters=None, ksize=3, activation="relu"):
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


class EarlyFusion(Layer):
    """
    A wrapper over TimeDistributed layer for Early Fusion.
    Returns a TimeDistributed Conv2d layer you can add to the model.

    <https://www.tensorflow.org/tutorials/eager/custom_layers#implementing_custom_layers>
    """
    def __init__(self, timeframes, iheight, iwidth, channels=3,
                nfilters=None, ksize=3, activation=linear):
        """
        :param timeframes: Number of timesteps/frames; the first index AFTER batch.
        :param iheight: Height of frame.
        :param iwidth: Width of frame.
        :param channels: Image channels.
        :param nfilters: Filters for Conv2d; will be imputed if None.
        :param ksize: Kernel size for Conv2d.
        :param activation: Activation for Conv2d.
        :return: A TimeDistributed Conv2d layer you can add to the model.
        """
        super(EarlyFusion, self).__init__()

        if nfilters is None:
            nfilters = 24 // timeframes

        self.convTd = TimeDistributed(
            Conv2D(nfilters, ksize, activation=activation, padding="same"),
            input_shape=(timeframes, iheight, iwidth, channels)
        )

    def call(self, inputs):
        x = self.convTd(inputs)
        x = tf.reduce_sum(x, axis=1)
        return x


if __name__ == '__main__':
    tf.enable_eager_execution()
    print "TF Executing Eagerly?", tf.executing_eagerly()

    x = tf.constant(np.ones((5, 3, 32, 32, 3)))
    print 'x', x.shape

    y = EarlyFusion(
        timeframes=3, iheight=32, iwidth=32, channels=3, nfilters=12
    )(x)

    print 'y', y.shape