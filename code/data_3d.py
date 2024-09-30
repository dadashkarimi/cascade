import numpy as np
import math
import os
from random import randint
from PIL import Image

import param_3d

import util

def load_db_detect_train_from_generator(x, y, dim):
    print("Loading positive training db from generator...")

    pos_db_12 = []
    pos_db_24 = []
    pos_db_48 = []
    # frame_dim = param_3d.framedim
    brain_indices = np.where(y > 0)
    if len(brain_indices[0]) == 0:
        print("No brain detected.")
        return 

    xmin = np.min(brain_indices[2])
    ymin = np.min(brain_indices[1])
    zmin = np.min(brain_indices[0])
    xmax = np.max(brain_indices[2])
    ymax = np.max(brain_indices[1])
    zmax = np.max(brain_indices[0])
    
    for z in range(zmin, zmax + 1):
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                if y + param_3d.img_size_12 <= ymax and x + param_3d.img_size_12 <= xmax and z + param_3d.img_size_12 <= zmax:
                    box_12 = x, y, z, x + param_3d.img_size_12, y + param_3d.img_size_12, z + param_3d.img_size_12
                    pos_db_12.append(box_12)
                if y + param_3d.img_size_24 <= ymax and x + param_3d.img_size_24 <= xmax and z + param_3d.img_size_24 <= zmax:
                    box_24 = x, y, z, x + param_3d.img_size_24, y + param_3d.img_size_24, z + param_3d.img_size_24
                    pos_db_24.append(box_24)
                if y + param_3d.img_size_48 <= ymax and x + param_3d.img_size_48 <= xmax and z + param_3d.img_size_48 <= zmax:
                    box_48 = x, y, z, x + param_3d.img_size_48, y + param_3d.img_size_48, z + param_3d.img_size_48
                    pos_db_48.append(box_48)
    
    return [pos_db_12, pos_db_24, pos_db_48]
    
def load_db_calib_train_from_generator(pos_db, ground_truth_mask, dim=12, iou_threshold=0.5):
    frame_dim = param_3d.framedim
    cali_scale = param_3d.cali_scale
    cali_off_x = param_3d.cali_off_x
    cali_off_y = param_3d.cali_off_y
    cali_off_z = param_3d.cali_off_z
    cali_patt_num = len(cali_scale) * len(cali_off_x) * len(cali_off_y) * len(cali_off_z)

    x_db_list_12 = [None for _ in range(cali_patt_num)]
    x_db_list_24 = [None for _ in range(cali_patt_num)]
    x_db_list_48 = [None for _ in range(cali_patt_num)]

    xmin, ymin, zmin, xmax, ymax, zmax = pos_db
    x_length = xmax - xmin
    y_length = ymax - ymin
    z_length = zmax - zmin

    for si, s in enumerate(cali_scale):
        for xi, x in enumerate(cali_off_x):
            for yi, y in enumerate(cali_off_y):
                for zi, z in enumerate(cali_off_z):
                    new_xmin = xmin - int(x * x_length / s)
                    new_ymin = ymin - int(y * y_length / s)
                    new_zmin = zmin - int(z * z_length / s)
                    new_xmax = new_xmin + int(x_length / s)
                    new_ymax = new_ymin + int(y_length / s)
                    new_zmax = new_zmin + int(z_length / s)

                    if (new_xmin < 0 or new_ymin < 0 or new_zmin < 0 or 
                        new_xmax >= frame_dim or new_ymax >= frame_dim or new_zmax >= frame_dim):
                        continue

                    window = (new_xmin, new_ymin, new_zmin, new_xmax, new_ymax, new_zmax)
                    iou = calculate_iou(window, ground_truth_mask)
                    # print(iou)
                    if iou < iou_threshold:
                        continue

                    calib_idx = (si * len(cali_off_x) * len(cali_off_y) * len(cali_off_z) +
                                 xi * len(cali_off_y) * len(cali_off_z) +
                                 yi * len(cali_off_z) +
                                 zi)
                    if dim == param_3d.img_size_12:
                        x_db_list_12[calib_idx] = window
                    elif dim == param_3d.img_size_24:
                        x_db_list_24[calib_idx] = window
                    elif dim == param_3d.img_size_48:
                        x_db_list_48[calib_idx] = window

    if dim == param_3d.img_size_12:
        return x_db_list_12
    elif dim == param_3d.img_size_24:
        return x_db_list_24
    else:
        return x_db_list_48