#!/usr/bin/env python
"""
created at: 23/04/19 1:15 AM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com), Akshay Chawla (akshaych@andrew.cmu.edu)

Join frames into a video using ffmpeg.

HOW TO USE? 

python join_frames.py <path/to/video.mp4> 

This will then create the follwing directories in your current folder
./temp
./temp/ip_frames (will use ffmpeg to convert video to frames in this directory)
./temp/op_frames (will upscale the frames from ip_frames (first downscaled) 
                    using model and put the resulting hr frames here )

After this utility has run cd to ./temp/op_frames and run the following command 
to convert images "frames___x.png" to a video
ffmpeg -i frame___%d.png video.mp4

"""

import tensorflow as tf 
from src_tensorflow.model import ENHANCE
import os, sys, time
import shutil
import numpy as np 
from PIL import Image


def im2patches(pil_img):
    """
    split a pil image into non-overlapping patches of 96x96 each
    """
    np_img = np.asarray(pil_img)
    patches = []
    for r_idx in range(np_img.shape[0] // 96):
        for c_idx in range(np_img.shape[1] // 96):
            patch = np_img[ r_idx*96:r_idx*96 + 96, c_idx*96:c_idx*96+96, : ]
            patches.append(patch)
    return patches 

def patches2im(patches):
    """
    reconstruct a 1080x1920x3 numpy image from 220x96x96x3 patches 
    """
    np_img = np.zeros((1080, 1920, 3), dtype=np.float)
    patch_idx = 0
    for r_idx in range(np_img.shape[0] // 96):
        for c_idx in range(np_img.shape[1] // 96):
            np_img[ r_idx*96:r_idx*96 + 96, c_idx*96:c_idx*96+96, : ] += patches[patch_idx]
            patch_idx += 1
    return np_img

def sanitize_img(img):
    """
    scale between 0-255 so that it is a regular image 
    """
    img_copy = np.copy(img)
    img_copy -= np.min(img_copy)
    img_copy /= np.max(img_copy)
    img_copy = img_copy * 255.0 
    return img_copy.astype(np.uint8)

def downsample_patches(patches):
    down_patches = [] 
    for p in patches: 
        im = Image.fromarray(p)
        im = im.resize((32,32), Image.BICUBIC)
        im_np = np.asarray(im)
        down_patches.append(im_np)
    return down_patches

def dummy_upsample(hr_frame):
    assert hr_frame.shape==(1,3,32,32,3)
    x = hr_frame[0,1].astype(np.uint8) 
    x_up = np.array(Image.fromarray(x).resize((96,96), Image.NEAREST))
    return np.expand_dims(x_up, 0)


def run():
    print "TF Executing Eagerly?", tf.executing_eagerly()

    # Instantiate model.
    from src_tensorflow.model import ENHANCE
    model = ENHANCE()
    optimizer = tf.optimizers.SGD(nesterov=True, momentum=0.9)

    # Restore checkpoint.
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_dir, 'data', 'checkpoints', 'host_biometrics')
    ckpt.restore(tf.train.latest_checkpoint(checkpoints_dir))

    # extract video frames
    video_path = sys.argv[1] 
    print("Extracting video frames for video {}".format(video_path))
    if os.path.exists("./temp"):
        shutil.rmtree("./temp")
    os.mkdir("./temp")
    os.mkdir("./temp/ip_frames")
    os.mkdir("./temp/op_frames")
    cmd = "ffmpeg -i {} {}___%d.png".format(video_path, "./temp/ip_frames/frame")
    os.system(cmd)

    num_frames = os.listdir("./temp/ip_frames")
    num_frames = [f.replace(".png","") for f in num_frames]
    num_frames = [int(f.split("___")[-1]) for f in num_frames]
    num_frames = max(num_frames)
    print("We have {} frames".format(num_frames))

    for frame_idx in range(1, num_frames+1 - 2):

        print("Processing for frame: {}".format(frame_idx))

        f1 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx)).convert("RGB")
        f2 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx+1)).convert("RGB")
        f3 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx+2)).convert("RGB")
        f1_patches = im2patches(f1)
        f2_patches = im2patches(f2)
        f3_patches = im2patches(f3)
        f1_lr_patches = downsample_patches(f1_patches)
        f2_lr_patches = downsample_patches(f2_patches)
        f3_lr_patches = downsample_patches(f3_patches)
        f1_lr_patches = np.array(f1_lr_patches) 
        f2_lr_patches = np.array(f2_lr_patches) 
        f3_lr_patches = np.array(f3_lr_patches) 
        
        frames = np.stack(
            (f1_lr_patches, f2_lr_patches, f3_lr_patches), axis=1
        )
        frames = frames.astype(np.float32)

        # predict  
        hr_frames = []
        for patch_idx in range(len(frames)):
            hr_frame, _, _ = model(frames[patch_idx: patch_idx+1])
            hr_frames.append(hr_frame.numpy())
            # hr_frame = dummy_upsample(frames[patch_idx: patch_idx+1])
            # hr_frames.append(hr_frame)

        hr_frames = np.concatenate(hr_frames, axis=0)

        hr_image = patches2im(hr_frames)
        hr_image = sanitize_img(hr_image)
    
        hr_image = Image.fromarray(hr_image)
        hr_image.save("./temp/op_frames/frame___{}.png".format(frame_idx))






        


if __name__ == "__main__":
    run()