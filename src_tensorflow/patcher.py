import numpy as np 
from PIL import Image

def im2patches(im, patch_size=(25,25), skip_last=False, zero_pad=True):
    """
    Converts an input image to patches. 
    """
    np_img = np.asarray(im).astype(np.float) # in case this is an PIL image 
    h,w,_ = np_img.shape
    patches = [] 
    ph, pw = patch_size
    _rows = h // ph + (0 if skip_last else 1)
    _cols = w // pw + (0 if skip_last else 1) 
    for r_idx in range(_rows): 
        for c_idx in range(_cols): 
            patch = np_img[r_idx*ph: r_idx*ph + ph, c_idx*pw: c_idx*pw + pw, :] 
            if (tuple(patch.shape[0:2]) != patch_size) and (zero_pad==True): 
                # this means we have hit the last patch, so zero pad 
                pad = ( (0, ph-patch.shape[0]), (0, pw-patch.shape[1]), (0,0) ) 
                expanded = np.pad(patch, pad, "constant", constant_values=0)
                patch = expanded
            patches.append(patch)
    return patches

def patches2im(patches, img_size=(127,127), skip_last=False, zero_pad=True):
    """
    Convert input patches to full image using img_size
    """
    patch_size = patches[0].shape[0:2] 
    ph, pw = patch_size 
    h,w = img_size 
    _rows = h // ph + (0 if skip_last else 1)
    _cols = w // pw + (0 if skip_last else 1) 
    np_img = np.zeros((ph*_rows, ph*_cols, 3)).astype(np.float)
    patch_idx = 0
    for r_idx in range(_rows): 
        for c_idx in range(_cols): 
            try:
                np_img[r_idx*ph: r_idx*ph + ph, c_idx*pw: c_idx*pw + pw, :] += patches[patch_idx] 
            except:
                import pdb; pdb.set_trace()
                print("error...")
            patch_idx += 1

    # chop off the excess zero_padded region 
    np_img = np_img[0:h, 0:w] 
    return np_img



def test(): 
    print("Running patcher self test...") 
    for i in range(10):
        im_h, im_w = np.random.randint(300, 400, size=(2,))
        ph, pw = np.random.randint(50, 150, size=(2,))
        im = np.random.randn(im_h,im_w,3) 
        patch_size = (ph,pw) 
        patches = im2patches(im, patch_size, skip_last=False, zero_pad=True)
        import pdb; pdb.set_trace()
        reconstructed = patches2im(patches, img_size=(im_h,im_w), skip_last=False, zero_pad=True)
        print("Image size: {} | patch_size: {} | recons OK? {}".format(
            (im_h, im_w), patch_size,  np.allclose(reconstructed, im)
        ))
    
if __name__ == "__main__":
    test()
    

    


