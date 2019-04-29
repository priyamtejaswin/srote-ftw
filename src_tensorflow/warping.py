import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from fusion import EarlyFusion
from tensorflow.keras import layers

tf.enable_eager_execution()


def coarse_flow(frames): 

    ef = EarlyFusion(timeframes=2, iheight=32, iwidth=32, channels=3)(frames)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(ef)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")(out)

    # subpixel conv upscaling x4 
    out = tf.depth_to_space(out, 4)

    return ef,out 

def fine_flow(early_fused, coarse_flow_vectors, coarse_compensated_frame): 

    fine_input = tf.concat((early_fused, coarse_flow_vectors, coarse_compensated_frame), axis=-1)
    out = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu", padding="same")(fine_input)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(out)
    out = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")(out)
    # subpixel conv upscaling x2 
    out = tf.depth_to_space(out, 2)

    return out

def test_flow():

    import ipdb; ipdb.set_trace()
    frames = np.random.randn(1,2,32,32,3) # batch x (I_t, I_t+1) x height x width x channels(RGB)
    frames = frames.astype(np.float32)

    # coarse flow
    early_fused, coarse_flow_vectors = coarse_flow(frames)

    # warp 
    next_frame = frames[:,1,:,:,:]
    coarse_compensated_frame = tf.contrib.image.dense_image_warp(next_frame, coarse_flow_vectors)

    # fine flow
    fine_flow_vectors = fine_flow(early_fused, coarse_flow_vectors, coarse_compensated_frame)

    # combine coarse_flow_vectors and fine_flow_vectors to get the final total_flow_vectors
    total_flow = coarse_flow_vectors + fine_flow_vectors

    # use total_flow to compensate i_t+1 back to i_t 
    compensated_frame = tf.contrib.image.dense_image_warp(next_frame, total_flow)

    print(compensated_frame.shape)



def test_warping():
    x = np.random.randn(4,4)
    y = np.zeros_like(x) 
    y[:, 1:] += x[:, :3]
    flow = np.zeros(shape=(1,4,4,2))
    flow[0,:,:,1] += 1
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=3)
    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=3)

    print(x.shape)
    print(y.shape)

    _y = tf.contrib.image.dense_image_warp(x, flow)
    _y = _y.numpy()

    print(_y.shape)

    # checking if my intution for warping is correct 
    same = np.allclose(_y[0,:,1:,0], y[0,:,1:,0])
    print("Intuition correct? {}".format(same))

    import ipdb; ipdb.set_trace()

    print("..done")

if __name__ == "__main__":
    test_flow()