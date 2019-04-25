import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

tf.enable_eager_execution()

x = np.random.randn(4,4)
y = np.zeros_like(x) 
y[:, 1:] += x[:, :3]
flow = np.zeros(shape=(1,4,4,2))
flow[0,:,:,0] += 1
x = np.expand_dims(x, axis=0)
x = np.expand_dims(x, axis=3)
y = np.expand_dims(y, axis=0)
y = np.expand_dims(y, axis=3)

print(x.shape)
print(y.shape)

_y = tf.contrib.image.dense_image_warp(x, flow)
_y = _y.numpy()

print(_y.shape)

import ipdb; ipdb.set_trace()

print("..done")