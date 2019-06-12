"""
This script just tests "if" mocomp works 
"""
import numpy as np 
import tensorflow as tf

import os
import sys

from src_tensorflow.model import ENHANCE
from src_tensorflow.utils import load_fnames, build_dataset
from src_tensorflow.patcher import im2patches, patches2im
from PIL import Image
from tqdm import tqdm
import shutil

def np2PIL(arr): 
    """ convert a numpy array to PIL image """ 
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr * 255.0 
    arr = arr.astype(np.uint8)
    pil_im = Image.fromarray(arr)
    return pil_im

def downsample_patches(patches):
    down_patches = [] 
    for p in patches: 
        im = Image.fromarray(p.astype(np.uint8)) 
        im = im.resize((32,32), Image.BICUBIC) 
        im_np = np.asarray(im).astype(np.float) 
        down_patches.append(im_np)
    return down_patches
    

# def im2patches(pil_img):
#     """
#     split a pil image into non-overlapping patches of 96x96 each
#     """
#     np_img = np.asarray(pil_img)
#     patches = []
#     for r_idx in range(np_img.shape[0] // 96):
#         for c_idx in range(np_img.shape[1] // 96):
#             patch = np_img[ r_idx*96:r_idx*96 + 96, c_idx*96:c_idx*96+96, : ]
#             patches.append(patch)
#     return patches 


# def patches2im(patches, full_res):
#     """
#     Reconstruct the `full_res` image from smaller `patches`.
#     :param patches: Super-resolved image patches.
#     :param full_res: Shape (h, w, c) of the full image.
#     :return:
#     """
#     np_img = np.zeros(full_res, dtype=np.float)
#     patch_idx = 0
#     for r_idx in range(np_img.shape[0] // 96):
#         for c_idx in range(np_img.shape[1] // 96):
#             np_img[ r_idx*96:r_idx*96 + 96, c_idx*96:c_idx*96+96, : ] += patches[patch_idx]
#             patch_idx += 1
#     return np_img

def test():
    print "TF Executing Eagerly?", tf.executing_eagerly()

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

    # check point directory 
    checkpoints_dir = sys.argv[2]
    assert os.path.isdir(checkpoints_dir), "2nd arg `checkpoints_dir` does not exist or is not a directory!"

    # Model prep.
    model = ENHANCE()
    optimizer = tf.optimizers.SGD(nesterov=True, momentum=0.9)

    # Restore checkpoint.
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(checkpoints_dir))


    for frame_idx in tqdm(range(1, num_frames+1 - 2)):

        f1 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx)).convert("RGB")
        f2 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx+1)).convert("RGB")
        f3 = Image.open("./temp/ip_frames/frame___{}.png".format(frame_idx+2)).convert("RGB")

        f1_patches = im2patches(np.asarray(f1), patch_size=(96,96), skip_last=False, zero_pad=True)
        f2_patches = im2patches(np.asarray(f2), patch_size=(96,96), skip_last=False, zero_pad=True)
        f3_patches = im2patches(np.asarray(f3), patch_size=(96,96), skip_last=False, zero_pad=True)

        # downsample the patches
        # TODO
        f1_ds_patches = downsample_patches(f1_patches) 
        f2_ds_patches = downsample_patches(f2_patches) 
        f3_ds_patches = downsample_patches(f3_patches) 

        f1_patches = np.array(f1_ds_patches) 
        f2_patches = np.array(f2_ds_patches) 
        f3_patches = np.array(f3_ds_patches)


        frames = np.stack(
            (f1_patches, f2_patches, f3_patches), axis=1
        )
        frames = frames.astype(np.float32) / 255.0

        # batched prediction 
        ops = []
        batch_size = 16 
        num_samples = frames.shape[0] 
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, start_idx+batch_size)
            # print start_idx, end_idx, num_samples
            # comp1, flow1 = model(frames[start_idx: end_idx])
            hr_frame, _, _ = model(frames[start_idx: end_idx])
            ops.append(hr_frame)
        ops = np.concatenate(ops, axis=0)
        ops = list(ops) # converts just the first arr dimension to list (as required by patches2im)

        assert len(ops) == frames.shape[0], "frames and ops have different shapes"
        op_shape = list(f1.size) 
        op_shape.reverse() 
        full_img = patches2im(ops, op_shape, skip_last=False, zero_pad=True)

        # meta data about a single image 
        print "min {} | max {} ".format(full_img.min(), full_img.max())

        # fix full_img 
        full_img = full_img - full_img.min() # range 0 to some max value
        full_img /= full_img.max()  # range 0 to 1


        # convert to PIL image and save 
        full_img = Image.fromarray((full_img * 255.0).astype(np.uint8))
        full_img.save("./temp/op_frames/frame___{}.png".format(frame_idx))


    # Join with ffmpeg.
    # print "Joining frames..."
    # join_cmd = "ffmpeg -i {}___%d.png {}.mp4".format("./temp/op_frames/frame", "./temp/result")
    # os.system(join_cmd)
    # print "Successfully executed command: %s"%join_cmd



        


    # for ix, (x,y) in enumerate(dataset):
    #     comp1, flow1 = model(x)

    #     import ipdb; ipdb.set_trace()
    #     reference_frame = x.numpy()[0,0]
    #     other_frame = x.numpy()[0,1]
    #     compensated_frame = comp1.numpy()[0]

    #     reference_frame = np2PIL(reference_frame)
    #     other_frame = np2PIL(other_frame)
    #     compensated_frame = np2PIL(compensated_frame)

    #     reference_frame.save("ref.png")
    #     other_frame.save("other.png")
    #     compensated_frame.save("comp.png")





    

if __name__ == "__main__":
    test()
