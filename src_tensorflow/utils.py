#!/usr/bin/env python
"""
created at: 23/04/19 1:20 AM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Utility functions. Data, io, processing, etc.
"""

import os
import sys
import tensorflow as tf


VFDEL = '___' ## VideoName___FrameId.png delimiter.
NFRAMES = 2 ## Number of consecutive frames considered per sample.

KERNEL = 192 ## Patch height and width.
STRIDE = 14 ## Stride while taking patches.
DOWNK = 96 ## Downscaled height and width.
BATCHSIZE = 64


def _fskey(f):
    vname, fid = f.rstrip('.png').split(VFDEL)
    fid = int(fid)
    return vname, fid


def load_fnames(fdir):
    all_frames = []

    vdirs = [os.path.join(fdir, d) for d in os.listdir(fdir)]
    vdirs = [d for d in vdirs if os.path.isdir(d)]
    print "Found %d vid files."%len(vdirs)

    for dpath in vdirs:
        print "In %s"%dpath
        frames = [f for f in os.listdir(dpath) if f.endswith('.png')]
        print "\tFound %d frames."%len(frames)

        frames = sorted(frames, key=lambda x:_fskey(x))
        frames = [os.path.join(dpath, f) for f in frames]

        ## Group sorted frames by NFRAMES.
        motion_frames = []
        for i in range(len(frames) - NFRAMES + 1):
            motion_frames.append(tuple(frames[i : i+NFRAMES]))

        ## Add to master list.
        all_frames.extend(motion_frames)

    print "Found %d frame groups of length %d."%(len(all_frames), NFRAMES)
    return all_frames


def load_image(fpath):
    image_string = tf.read_file(fpath)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def make_patches(image):
    channels = tf.shape(image)[-1]
    patches = tf.image.extract_image_patches(
        image,
        ksizes=[1, KERNEL, KERNEL, 1],
        strides=[1, STRIDE, STRIDE, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.squeeze(patches)

    shape = tf.shape(patches)  ## [frames, patches_x, patches_y, `channels`]

    patches = tf.reshape(patches, [NFRAMES, shape[1], shape[2], KERNEL, KERNEL, channels])
    patches = tf.reshape(patches, [NFRAMES, shape[1] * shape[2], KERNEL, KERNEL, channels])

    patches = tf.transpose(patches, [1, 0, 2, 3, 4])
    return patches


def make_xy(patches):
    downed = tf.image.resize_images(patches, [DOWNK, DOWNK])
    return downed, patches


if __name__ == '__main__':
    dirpath = sys.argv[1]
    load_fnames(dirpath)