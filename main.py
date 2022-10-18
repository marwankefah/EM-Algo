import copy
import os
from pathlib import Path

import pylab
import logging
import sys

from metrics import dice_coef_multilabel

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from EM import EM
import matplotlib.pyplot as plt
import numpy as np
from utils import read_img, flatten_img

data_path = Path('./data')
data_folders = os.listdir(data_path)

init_type = 'kmeans'
n_clusters = 3
max_iter = 50
n_clusters = 3
error = 0.1

# 0, 1 = red (CSF), 2="Grey matter", 3= "WM"

for data_folder in data_folders:
    T1_fileName = data_path / data_folder / Path('T1.nii')
    T2_fileName = data_path / data_folder / Path('T2_FLAIR.nii')
    gt_fileName = data_path / data_folder / Path('LabelsForTesting.nii')

    T1 = read_img(filename=T1_fileName)
    T2 = read_img(filename=T2_fileName)
    gt = read_img(filename=gt_fileName)

    gt_mask = copy.deepcopy(gt)
    gt_mask[gt_mask > 0] = 1
    T1_masked = np.multiply(T1, gt_mask)
    T2_masked = np.multiply(T2, gt_mask)
    x, y, z = T1.shape

    stacked_features = np.stack((flatten_img(T1_masked, mode='3d'), flatten_img(T2_masked, mode='3d')), axis=1)

    em = EM(stacked_features.squeeze(), init_type, n_clusters)
    recovered_img = em.execute(error, max_iter, visualize=False)

    mean_values = np.unique(recovered_img)
    seg_mask = np.zeros_like(recovered_img)

    for i in range(n_clusters + 1):
        seg_mask[recovered_img == mean_values[i]] = i

    dice_list = dice_coef_multilabel(gt, seg_mask)
    print(dice_list)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(T1_masked[:, :, 24])
    # plt.show()
    ax1.set_title('T1 image')

    ax2.imshow(gt[:, :, 24], cmap=pylab.cm.cool)
    # plt.show()
    ax2.set_title('Ground Truth')
    ax3.imshow(recovered_img[:, :, 24], cmap=pylab.cm.cool)
    ax3.set_title('Segmented Image')
    plt.title(T1_fileName)
    plt.tight_layout()
    plt.show()
