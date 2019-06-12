import numpy as np 
import os, sys, time, ipdb
from skimage import transform

def test_xy(): 
    x_1 = np.random.randn(96,96,3)
    tform = transform.SimilarityTransform(scale=1, rotation=0, translation=(0,1))
    x_2 = transform.warp(x_1, tform) 

    # x shape = (batchsize, 2, h, w, 3) 
    # y shape = (batchsize, h, w, 3) , which should be same as the reference image , since we're warping other --> reference 

    x_1 = np.expand_dims(x_1, 0) 
    x_2 = np.expand_dims(x_2, 0) 

    x = np.stack((x_1, x_2), axis=1) 
    y = x_1 

    return x.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    test_xy()



    


    
