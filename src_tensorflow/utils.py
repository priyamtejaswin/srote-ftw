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
    """
    Generate sorting key from a frame file name.

    :param f: VideoName___FrameId.png
    :return: (VideoName, int(FrameId))
    """
    vname, fid = f.rstrip('.png').split(VFDEL)
    fid = int(fid)
    return vname, fid


def load_fnames(fdir):
    """
    Load all frame names from every video directory.
    Group frames by NFRAMES param.

    :param fdir: Root directory, with all video directories, each containing video frames.
    :return: List of grouped frames.
    """
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
    """
    Read an image. Convert to tf.float32 ==> THIS IS IMPORTANT!!

    :param fpath: Path to frame image file.
    :return: Decoded image. (HEIGHT, WIDTH, CHANNEL)
    """
    image_string = tf.read_file(fpath)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def make_patches(image):
    """
    Extract patches from a grouped Tensor (NFRAMES, HEIGHT, WIDTH, CHANNELS).
    ==> Reshape patches to Tensor (NFRAMES, NUM_PATCHES, KERNEL, KERNEL, CHANNELS).
    ==> Swap axes -- aka numpy.transpose (NUM_PATCHES, NFRAMES, KERNEL, KERNEL, CHANNELS).
    This brings NUM_PATCHES outside.
    Check the README to understand how this works.

    :param image: Grouped Tensor (NFRAMES, HEIGHT, WIDTH, CHANNELS).
    :return: Patches Tensor (NUM_PATCHES, NFRAMES, KERNEL, KERNEL, CHANNELS).
    """
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
    """
    Downscale input and return (x, y) pairs.

    :param patches: Original high-res image.
    :return: Downsampled image. Kernel is default (BiLinear??)
    """
    downed = tf.image.resize_images(patches, [DOWNK, DOWNK])
    return downed, patches


def build_dataset(batched_fnames):
    """
    Build tf.data.Dataset from grouped frame paths.

    :param batched_fnames: List of frames, grouped by NFRAMES.
    :return: Proper tf.data.Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices(batched_fnames[:4])  ## Paired frames.
    dataset = dataset.shuffle(buffer_size=len(batched_fnames))  ## Paired frames are shuffled.

    ## Flatten everything | Order will be preserved in map and flat_map.
    ## https://stackoverflow.com/questions/49960875 -- flatten
    ## https://stackoverflow.com/questions/51015918 -- order
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    dataset = dataset.map(load_image)  ## Single frame images loaded.

    dataset = dataset.window(size=NFRAMES, drop_remainder=True)  ## Window consecutive frames.
    dataset = dataset.flat_map(lambda dset: dset.batch(NFRAMES))  ## Group the frames | Loaded frames are paired again.

    dataset = dataset.map(
        make_patches)  ## Generate patches for paired frames AND swap axes : [patches, NFRAMES, k, k, c]

    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)  ## Flatten | single tensors of [NFRAMES, k, k, c]

    dataset = dataset.shuffle(buffer_size= BATCHSIZE*3)  ## After single tensors and BEFORE (x,y) generation.

    dataset = dataset.map(make_xy)  ## Return (X,Y): (Downscaled, Original)

    dataset = dataset.batch(BATCHSIZE)  ## Final batching.
    return dataset


if __name__ == '__main__':
    dirpath = sys.argv[1]
    load_fnames(dirpath)