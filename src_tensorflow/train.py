#!/usr/bin/env python
"""
created at: 2019-05-07 11:32
created by: Priyam Tejaswin (tejaswin.p@flipkart.com), Akshay Chawla (akshaych@andrew.cmu.edu)

Main training file.

Steps before training:
1. Create a `videos` directory in the root folder of this project.
Dump all video files in this directory.

2. Use `dump_frames.py` file to process all videos.
This will create a `frames` directory.

3. Run `python train.py /path/to/frames_dir` to begin training.
"""

import tensorflow as tf
import os
import sys

from tqdm import tqdm
from model import ENHANCE
from losses import CombinedLoss
from utils import load_fnames, build_dataset


# tf.random.set_seed(1)
print 'TF version:', tf.__version__
print 'TF executing eagerly?', tf.executing_eagerly()

frames_dir = sys.argv[1]
if not os.path.isdir(frames_dir):
    raise OSError('Directory does not exist -- ' + frames_dir)

model = ENHANCE()
lossfn = CombinedLoss().all_loss
optimizer = tf.optimizers.SGD(nesterov=True, momentum=0.9)

batched_fnames = load_fnames(frames_dir)
dataset = build_dataset(batched_fnames)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds, (flow1, flow2), (comp1, comp2) = model(x)
        train_loss = lossfn(y_true=y, y_pred=preds,
                          flow1=flow1, flow2=flow2,
                          comp1=comp1, comp2=comp2,
                          frames=x)

    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss


template = 'Epoch: {}, Step: {}, Train Loss: {}'
for epoch in range(1):
    for ix, (x,y) in enumerate(dataset):
        train_loss = train_step(x, y)
        print template.format(epoch+1, ix+1, train_loss.numpy())