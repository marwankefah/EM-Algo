import copy

import numpy as np
import nibabel as nib
from scipy.ndimage._filters import gaussian_filter


def read_img(filename, blur_sigma=None):
    img_3d = nib.load(filename)
    affine = img_3d.affine
    img_3d = img_3d.get_fdata()
    if blur_sigma:
        img_3d = gaussian_filter(img_3d, blur_sigma)

    return img_3d,affine


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


def get_features(T1, T2, gt, use_T2):
    gt_mask = copy.deepcopy(gt)
    gt_mask[gt_mask > 0] = 1
    T1_masked = np.multiply(T1, gt_mask)
    # x, y, z = T1.shape
    T2_masked = None
    if use_T2:
        T2_masked = np.multiply(T2, gt_mask)
        stacked_features = np.stack((flatten_img(T1_masked, mode='3d'), flatten_img(T2_masked, mode='3d')),
                                    axis=1).squeeze()
    else:
        stacked_features = flatten_img(T1_masked, mode='3d')

    return stacked_features, T1_masked, T2_masked
