import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from fusion import EarlyFusion
from tensorflow.keras import layers


def coarse_flow(frames): 

    ef = EarlyFusion(timeframes=2, iheight=32, iwidth=32, channels=3)(frames)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(ef)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")(out)

    # subpixel conv upscaling x4 
    out = tf.depth_to_space(out, 4)

    return out 

def fine_flow(frames, coarse_flow_vectors, coarse_compensated_frame): 

    ef = EarlyFusion(timeframes=2, iheight=32, iwidth=32, channels=3)(frames)
    out = tf.concat((ef, coarse_flow_vectors, coarse_compensated_frame), axis=-1)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")(out)

    # subpixel conv upscaling x2 
    out = tf.depth_to_space(out, 2)

    return out

def motion_compensation(reference_frame, other_frame): 
    """
    use spatial transformer networks to compensate other_frame to reference_frame 
    reference_frame: batch x 32 x 32 x 3 
    other_frame: batch x 32 x 32 x 3 
    """ 
    frames = tf.stack((reference_frame, other_frame), axis=1)

    # coarse flow
    coarse_flow_vectors = coarse_flow(frames)

    # warp 
    coarse_compensated_frame = tf.contrib.image.dense_image_warp(other_frame, coarse_flow_vectors)

    # fine flow
    fine_flow_vectors = fine_flow(frames, coarse_flow_vectors, coarse_compensated_frame)

    # combine coarse_flow_vectors and fine_flow_vectors to get the final total_flow_vectors
    total_flow = coarse_flow_vectors + fine_flow_vectors

    # warp  
    compensated_frame = tf.contrib.image.dense_image_warp(other_frame, total_flow)

    return compensated_frame
    


def test_flow():

    import ipdb; ipdb.set_trace()
    reference_frame = np.random.randn(1,32,32,3).astype(np.float32)
    other_frame = np.random.randn(1,32,32,3).astype(np.float32)

    compensated_frame = motion_compensation(reference_frame, other_frame)

    print(compensated_frame.shape)


if __name__ == "__main__":
    tf.enable_eager_execution()
    test_flow()