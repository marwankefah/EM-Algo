import numpy as np
import nibabel as nib

def read_img(filename):
    img_3d = nib.load(filename)
    img_3d = img_3d.get_fdata()
    return img_3d
def flatten_img(img_3d, mode):
    if mode == '3d':
        x, y, z = img_3d.shape
        img_2d = img_3d.reshape(x * y * z, 1)
        img_2d = np.array(img_2d, dtype=np.float)
    elif mode == '2d':
        x, y = img_3d.shape
        img_2d = img_3d.reshape(x * y, 1)
        img_2d = np.array(img_2d, dtype=np.float)
    return img_2d