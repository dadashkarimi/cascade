from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full

import tensorflow.keras.layers as KL
import voxelmorph as vxm
from utils import *
from help import *

import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pathlib
import surfa as sf
import re
import json
from keras import backend as K
import param_3d
import data
import model_3d
from data_3d import *
import scipy.ndimage as ndimage

import nibabel as nib
from tqdm import tqdm
import scipy.ndimage as ndimage

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-model', '--model', choices=['b2','b3'], default='b2')
args = parser.parse_args()


if args.model=="b2":
    validation_folder_path="test/haste_trim2"
elif args.model=="b3":
    validation_folder_path="test/haste_trim3"



random=False
full_random=True

mom_ids , images, masks , b2_images, b3_images, b2_masks, b3_masks = load_val(validation_folder_path) 
image_index = 7
img = images[image_index]
mask=masks[image_index]




# model=models[3]

positions_48, indices_48 = generate_position_map((192,192,192), param_3d.img_size_48,64 )
positions_36, indices_36 = generate_position_map((192,192,192), param_3d.img_size_24, 64)
positions_24, indices_24 = generate_position_map((param_3d.img_size_48,)*3, param_3d.img_size_24, 32)
positions_12, indices_12 = generate_position_map((param_3d.img_size_24,)*3, param_3d.img_size_12, 32)
positions_6, indices_6 = generate_position_map((param_3d.img_size_12,)*3, param_3d.img_size_6, 32)

noise_model = get_model("noise")

subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
detection_results = []
    
for folder in subfolders:
    tf.keras.backend.clear_session()
    mom_str = folder.split("_")[1]
    match = re.search(r"__(\d+)wk__", folder)
    age = int(match.group(1))
    if mom_str.isdigit():
        mom = int(mom_str)
    else:
        mom = 0

    print("********** Mom:", mom)
    folder_path = os.path.join(validation_folder_path, folder)
    filename = os.path.join(folder_path, "image.nii.gz") if os.path.exists(os.path.join(folder_path, "image.nii.gz")) else os.path.join(folder_path, "image.mgz")

    mask_filename = os.path.join(folder_path, "manual.nii.gz")
    image = sf.load_volume(filename)
    
    new_voxsize = [dynamic_resize(image)]*3
        
    orig_voxsize = image.geom.voxsize
    crop_img = image.resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
    img = crop_img.resize(new_voxsize, method="linear").reshape([192, 192, 192])
    
    mask = sf.load_volume(mask_filename).resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
    mask = mask.resize(new_voxsize).reshape([192, 192, 192, 1])
    mask.data[mask.data != 0] = 1
    mask.data = find_largest_component(mask.data)


    

    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    pred_list = []
    detection_result = [-1, -1, -1, -1]
    

    try:
        
        # First stage
        detection, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192 = first_stage_prediction(img, mask, mom, noise_model, positions_48)
        
        
        if detection:
            pred_list.append(first_pred_192)
                
            
        if not detection:
                
            # 1/2 stage
            detection, pred_24, valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 = first_half_stage_prediction(img, mask, mom, positions_36)
            

            if not detection:
                raise ValueError("No mask found in first stage!")
            else:
                pred_list.append(first_pred_192)
             # Third stage
            detection, pred_12, valid_position_index_24, cube_12, mask_12, third_pred_192 = third_stage_prediction(img, mask, pred_24, cube_24, mask_24, pred_48, cube_48, mask_48, valid_position_index_48, valid_position_index_192,mom, positions_12)
            

            if not detection:
                raise ValueError("No mask found in third stage!")
            else:
                pred_list.append(third_pred_192)
            # continue
            
        else:
            pred_list.append(first_pred_192)
            
        print("len(pred_list)",len(pred_list))

        
        # # Second stage
        detection, pred_24, pred_48, valid_position_index_48, valid_position_index_192, cube_24, mask_24, second_pred_192 = second_stage_prediction(img, mask, pred_48, cube_48, mask_48, valid_position_index_192, mom, positions_24)

        if not detection:
            raise ValueError("No mask found in second stage!")
        else:
            pred_list.append(second_pred_192)
        # Third stage
        detection, pred_12, valid_position_index_24, cube_12, mask_12, third_pred_192 = third_stage_prediction(img, mask, pred_24, cube_24, mask_24, pred_48, cube_48, mask_48, valid_position_index_48, valid_position_index_192,mom, positions_12)
       

        if not detection:
            raise ValueError("No mask found in third stage!")
        else:
            pred_list.append(third_pred_192)
        
        # Fourth stage
        detection, pred_6, valid_position_index_12, cube_6, fourth_pred_192 = fourth_stage_prediction(img, mask, pred_12, cube_12, mask_12, cube_24, mask_24, cube_48, mask_48, valid_position_index_24, valid_position_index_48, valid_position_index_192, mom, positions_6)

        if not detection:
            raise ValueError("No mask found in fourth stage!")
        else:
            pred_list.append(fourth_pred_192)
    except ValueError as e:
        if len(pred_list) == 0:
            print("pred_list == 0")
            pred_list = [np.zeros((192,192,192))]

        print("len(pred_list)",len(pred_list))
        pred_192 = process_predictions(pred_list,param_3d)
        print("Consensus prediction broken.. ")
        print("Mom:",mom, " ### my_hard_dice: ", my_hard_dice(pred_192, mask.data))
        nib.save(nib.Nifti1Image(pred_192.astype(np.float32), np.array(img.geom.vox2world)), f"{folder_path}/cascade.nii.gz")
        continue  # Skip to the next 

    pred_192 = process_predictions(pred_list,param_3d)
        
    print("Consensus prediction .. ")
    print("Mom:",mom, " ### my_hard_dice: ", my_hard_dice(pred_192, mask.data))
    nib.save(nib.Nifti1Image(pred_192.astype(np.float32), np.array(img.geom.vox2world)), f"{folder_path}/cascade.nii.gz")
    