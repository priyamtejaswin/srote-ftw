#!/usr/bin/env python
"""
created at: 27/04/19 5:37 PM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

ESPCN introduces a very efficient way to upscale in image.
<https://arxiv.org/pdf/1609.05158.pdf>, Section 2.2

This involves 're-arranging' [B, H, W, C*k*k] to [B, H*k, W*k, C]
where `k` was the downscale factor. The transformation is not trivial.

<https://github.com/tetrachrome/subpixel> describes how to do this in Tf (phase_shift).

Turns out, there is a native Tf op `depth_to_space` which also does this.
<https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space>

This script simply shows the difference b/w the two.
Ideally, the two should be the same -- I think the `tf.split` behavior
leads to a slight difference. Transposing the `phase_shift` output
gives the `depth_to_space` output.

Script modified for tf-1.13.1 from <https://gist.github.com/poolio/cc887cf0f4db40160d9ff7b13086e8a5>
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm, figure, GridSpec, xticks, yticks, gca, subplot, imshow, axis, title


tf.enable_eager_execution()
print "Tf Executing Eagerly?", tf.executing_eagerly()


def _new_phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, axis=1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, [1]) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, b, axis=1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, [1]) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, axis=3)
        X = tf.concat([_new_phase_shift(x, r) for x in Xc], axis=3)
    else:
        X = _new_phase_shift(X, r)
    return X


# Feature map with shape [1, 8, 8, 4] with each feature map i having value i
x = np.ones((1, 8, 8, 4)) * np.arange(4)[None, None, None, :]
# Convert to a [1, 16, 16, 1] Tensor
y = tf.depth_to_space(tf.constant(x), 2)

# Plot results
figure(figsize=(12, 4.5))
gs = GridSpec(2, 5, width_ratios=[1, 1, 2, 2, 2])
for i in xrange(4):
    plt.subplot(gs[i // 2, i % 2])
    plt.imshow(x[:, :, :, i].squeeze(), cmap=cm.jet, vmin=0, vmax=4, interpolation='nearest')
    # Add ticks at pixels, annoyingly have to offset by 0.5 to line up with pixels
    xticks(0.5 + np.arange(8))
    yticks(0.5 + np.arange(8))
    plt.gca().set_xticklabels([])
    gca().set_yticklabels([])
    plt.title('feature %d' % i)

subplot(gs[:, 2])
print x.shape
out_ps = PS(tf.constant(x), 2)
imshow(tf.squeeze(out_ps).numpy(), cmap=cm.jet, vmin=0, vmax=4, interpolation='nearest')
axis('off')
title('phase shift')

subplot(gs[:, 3])
imshow(tf.squeeze(y), cmap=cm.jet, vmin=0, vmax=4, interpolation='nearest')
axis('off')
title('depth_to_space')

subplot(gs[:, 4])
imshow(tf.squeeze(out_ps).numpy().T, cmap=cm.jet, vmin=0, vmax=4, interpolation='nearest')
axis('off')
title('phase shift TRANSPOSED')

plt.gcf().tight_layout()
plt.show()