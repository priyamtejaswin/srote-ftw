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
from datetime import datetime
dt_start = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

from model import Mocomp_test
from losses import CombinedLoss
from utils import load_fnames, build_dataset
from tensorflow.keras.losses import mean_squared_error
from losses import huber_loss

from single_test import test_xy


# tf.random.set_seed(1)
print 'TF version:', tf.__version__
print 'TF executing eagerly?', tf.executing_eagerly()

frames_dir = sys.argv[1]
if not os.path.isdir(frames_dir):
    raise OSError('Directory does not exist -- ' + frames_dir)

# Model prep.
model = Mocomp_test()
lossfn = CombinedLoss().all_loss
optimizer = tf.keras.optimizers.SGD(nesterov=True, momentum=0.9)

# Dataset.
# batched_fnames = load_fnames(frames_dir, NFRAMES=2)
# dataset = build_dataset(batched_fnames, NFRAMES=2)
x_test, y_test = test_xy()

# Callbacks.
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
checkpoint_dir = os.path.join(project_dir, 'data', 'checkpoints', dt_start)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

logs_dir = os.path.join(project_dir, 'data', 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

train_summary_writer = tf.summary.create_file_writer(os.path.join(logs_dir, dt_start))

# Training function.
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        comp1, flow1 = model(x)

        # Staged MSE Calculation 
        # mse_loss = tf.reduce_sum( tf.square(y - comp1), axis=-1 ) # channels 
        # mse_loss = tf.reduce_sum( mse_loss, axis=-1) # width
        # mse_loss = tf.reduce_sum( mse_loss, axis=-1) # height
        # mse_loss = tf.reduce_mean( mse_loss ) # mean accross batch 
        mse_loss = tf.reduce_mean( tf.square( y - comp1) )

        # + huber loss 
        # hubloss = huber_loss(flow1) 
        hubloss = 0.0
        train_loss = mse_loss  #+ 0.01 * hubloss 

        
        # --------------- OLD LOSS CALCULATIONS ----------------
        # train_loss = tf.reduce_mean(tf.square(y-comp1))  

        # preds, (flow1, flow2), (comp1, comp2) = model(x)
        # train_loss = lossfn(y_true=y, y_pred=preds,
        #                   flow1=flow1, flow2=flow2,
        #                   comp1=comp1, comp2=comp2,
        #                   frames=x)

    # gradients = tape.gradient(train_loss, model.trainable_variables)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss, hubloss, mse_loss, flow1
    # return mse_loss

# Training loop.
template = 'Epoch: {}, Step: {}, Train Loss: {}'
global_step = 0

# keep aside these for loops for now
# for epoch in range(30):
    # for ix, (x,y) in enumerate(dataset):

for iter in range(5000):
    global_step += 1
    train_loss, hubloss, mse_loss, flow_vecs = train_step(x_test, y_test)
    # mse_loss = train_step(x,y) 
    
    # if True:
    if global_step % 10 == 0:
        # print template.format(epoch+1, ix+1, train_loss.numpy())
        print "iteration: {} | mse: {} | hublos: {} | trainloss: {}".format(
                iter+1, mse_loss,hubloss,train_loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.numpy(), step=global_step)
            tf.summary.scalar('hubloss', hubloss.numpy(), step=global_step)
            tf.summary.scalar('mse_loss', mse_loss.numpy(), step=global_step)
    
    
    if global_step % 50 == 0:
        print 'Saving checkpoint ...'
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_step_%d'%global_step)
        ckpt.save(file_prefix=checkpoint_prefix)

import ipdb; ipdb.set_trace()
print("finished..")

# test the output 
y_pred, _ = model(x_test) 


