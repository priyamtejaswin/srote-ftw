#!/usr/bin/env python
"""
created at: 01/05/19 10:20 PM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Custom layers for flow-estimation and motion-compensation.
Refer `src_tensorflow/warping.py`
"""


import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from fusion import EarlyFusion
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D


class CoarseFlow(Layer):
    """
    Coarse-Flow layer.
    """
    def __init__(self):
        super(CoarseFlow, self).__init__()

        self.Efl = EarlyFusion(
            timeframes=2, iheight=32, iwidth=32, channels=3
        )
        self.C1 = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")
        self.C2 = Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.C3 = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")
        self.C4 = Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")
        self.C5 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")

    def call(self, frames):
        ef = self.Efl(frames)
        out = self.C1(ef)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.C5(out)

        upped = tf.nn.depth_to_space(out, 4)
        return upped


class FineFlow(Layer):
    """
    Fine-Flow layer.
    """
    def __init__(self):
        super(FineFlow, self).__init__()

        self.Efl = EarlyFusion(
            timeframes=2, iheight=32, iwidth=32, channels=3
        )

        self.C1 = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu", padding="same")
        self.C2 = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")
        self.C3 = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")
        self.C4 = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same")
        self.C5 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="tanh", padding="same")

    def call(self, frames, coarse_flow_vectors, coarse_compensated_frame):
        """
        :param frames: Batch of frames.
        :param coarse_flow_vectors: Output from Coarse-Flow.
        :param coarse_compensated_frame: Motion compensated frame after Coarse-Flow.
        """

        ef = self.Efl(frames)
        out = tf.concat((ef, coarse_flow_vectors, coarse_compensated_frame), axis=-1)

        out = self.C1(out)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.C5(out)

        upped = tf.nn.depth_to_space(out, 2)
        return upped


class MotionCompensation(Layer):
    """
    Motion-Compensation layer.
    Uses Coarse-Flow and Fine-Flow instances.
    """
    def __init__(self):
        super(MotionCompensation, self).__init__()

        self.coarse_flow = CoarseFlow()
        self.fine_flow = FineFlow()

    def call(self, reference_frame, other_frame):
        """
        Both are of size (batch, 32, 32, 3)

        :param reference_frame: Source.
        :param other_frame: Make 'sans' motion.
        """
        frames = tf.stack((reference_frame, other_frame), axis=1)

        # coarse flow
        coarse_flow_vectors = self.coarse_flow(frames)

        # warp
        coarse_compensated_frame = tfa.image.dense_image_warp(other_frame, coarse_flow_vectors)

        # fine flow
        fine_flow_vectors = self.fine_flow(frames, coarse_flow_vectors, coarse_compensated_frame)

        # combine coarse_flow_vectors and fine_flow_vectors to get the final total_flow_vectors
        total_flow = coarse_flow_vectors + fine_flow_vectors

        # warp
        compensated_frame = tfa.image.dense_image_warp(other_frame, total_flow)

        return compensated_frame, total_flow  # Needed for Huber loss computation.


if __name__ == '__main__':
    print 'Testing...'
    tf.enable_eager_execution()
    print "TF Executing Eagerly?", tf.executing_eagerly()

    x = tf.constant(np.random.rand(5, 2, 32, 32, 3).astype(np.float32))

    print 'Testing EarlyFusion...'
    efl = EarlyFusion(timeframes=2, iheight=32, iwidth=32, channels=3)
    print efl(x).shape

    print 'Testing CoarseFlow...'
    cf = CoarseFlow()
    print cf(x).shape

    print 'Testing MC...'
    mc = MotionCompensation()
    compensated_frame, total_flow = mc(x[:, 0], x[:, 1])
    print compensated_frame.shape, total_flow.shape