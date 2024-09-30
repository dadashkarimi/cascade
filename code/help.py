from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite.tf.utils.augment import draw_perlin_full
import voxelmorph as vxm
import os
import glob
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
import surfa as sf
import math
# import Image
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian
import param_3d
import model_3d
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from utils import find_bounding_box, find_random_bounding_box, apply_gaussian_smoothing, extract_cube
from tensorflow.keras.models import Model
import neurite as ne
from utils import find_largest_component
import scipy.ndimage as ndimage
from utils import my_hard_dice
from utils import *
from tqdm import tqdm
from model_3d import noiseModel

models=["6Net","12Net","24Net","48Net","48PerlinNet","gmm"]

def get_cube_and_model(model,img, mask, random,full_random,trimester):
    epsilon =1e-7
    tfolder = "models_b"+str(trimester)
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    if model=="6Net":
        print("6Net model is loading")
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        input_img = Input(shape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_6)
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_6,full_random=full_random)
            
    
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_6Shot", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)
    
        
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2)
        new_mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_6)

    elif model=="12Net":
        print("12Net model is loading")
    
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_12)
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_12,margin=param_3d.img_size_12 , full_random=full_random)
    
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_12Shot", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_12)
        new_mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)

    elif model=="24Net":
        print("24Net model is loading")
    
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_24)
        
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_24,margin=param_3d.img_size_24,full_random=full_random)
    
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_24Shot", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24)
        print(x1, y1, z1, x2, y2, z2)
        new_mask = extract_cube(mask, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
  
    elif model=="48Net":
        print("48Net model is loading")
        
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_48)
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_48,full_random=full_random)
    
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_48net_elipses", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48)
        new_mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_48)

    elif model=="gmm":
        print("gmm model is loading")
        
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_48)
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_48,full_random=full_random)
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_gmm", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        # input_img =  apply_gaussian_smoothing(input_img,sigma = 1.0,kernel_size = 3)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48)
        new_mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_48)
    elif model=="noise":
        print("noise model is loading")
        latest_weight = max(glob.glob(os.path.join("models_cascade_noise_net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_192)
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        noise_model = noiseModel((param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        generated_noise_img = min_max_norm(input_img)
        
        denoised_img = noise_model(generated_noise_img)
        
        combined_model = Model(inputs=input_img, outputs=denoised_img)
        combined_model.load_weights(latest_weight)    
        
        cube = img.data 
        new_mask = mask.data
        
        
    box = np.zeros((192, 192, 192), dtype=int)
    box[x1:x2+1, y1:y2+1, z1:z2+1] = 1
    return cube , new_mask, box, combined_model

def get_model(model,trimester):
    epsilon =1e-7
    tfolder = "models_b"+str(trimester)
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    if model=="6Net":
        print("6Net model is loading")
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        input_img = Input(shape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
            
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_6net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)
    elif model=="12Net":
        print("12Net model is loading")
    
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
    
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_12net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
    elif model=="24Net":
        print("24Net model is loading")
    
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        latest_weight = max(glob.glob(os.path.join(tfolder,"models_cascade_24net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)            
    elif model=="48Net":
        print("48Net model is loading")
        
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')

        model_dir = "models_cascade_48net_elipses"
        latest_weight = max(glob.glob(os.path.join(tfolder,model_dir, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        input_img =  apply_gaussian_smoothing(input_img,sigma = 1.0,kernel_size = 3)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)   
        
    elif model=="48PerlinNet":
        print("48Net model is loading")
        
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')

        model_dir = "models_cascade_48net_bones"
        latest_weight = max(glob.glob(os.path.join(model_dir, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        input_img =  apply_gaussian_smoothing(input_img,sigma = 1.0,kernel_size = 3)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight) 
        
    elif model=="gmm":
        print("gmm model is loading")
        
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_gmm", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        input_img =  apply_gaussian_smoothing(input_img,sigma = 1.0,kernel_size = 3)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)  
    elif model=="noise":
        print("noise model is loading")
        latest_weight = max(glob.glob(os.path.join("models_cascade_noise_net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        noise_model = noiseModel((param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        generated_noise_img = min_max_norm(input_img)
        
        denoised_img = noise_model(generated_noise_img)
        
        combined_model = Model(inputs=input_img, outputs=denoised_img)
        combined_model.load_weights(latest_weight)
        
        
    return combined_model


def get_min_max_size(level, age):
    sizes = {
        1: {'min': 5000, 'max': 52278},
        2: {'min': 5000, 'max': 52278},
        3: {'min': 5000, 'max': 52278},
        4: {'min': 5000, 'max': 52278},
    }

    if level not in sizes:
        raise ValueError(f"Invalid level: {level}. Valid levels are {list(sizes.keys())}")

    size_dict = sizes[level]
    min_size = size_dict['min'] if age < 26 else size_dict['min']# + 5000
    max_size = size_dict['max'] if age > 26 else size_dict['max'] #- 20000
    return min_size , max_size
    
def first_stage_prediction(img, mask, mom, positions_48, validation_path,trimester):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    touches_edge = 0
    pred_48 = np.zeros((param_3d.img_size_48,)*3)
    pred_24 = np.zeros((param_3d.img_size_24,)*3)
    first_pred_192 = np.zeros((param_3d.img_size_192,)*3)
    valid_position_index_192 = None
    cube_48 = None
    mask_48 = None
    if trimester == 3:
        combined_model_48 = get_model(models[4],trimester)
    else:
        print("48 model for trimester 2 is being loaded")
        max_size = 60000
        combined_model_48 = get_model(models[3],trimester) 

    for min_size in tqdm(range(min_size, 8000, -5000)):
        detection, touches_edge, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192 = find_brain_48(positions_48, min_size, max_size, combined_model_48, img, mask , validation_path,trimester)
        if detection:
            break
    
    if not detection:
        combined_model_48 = get_model(models[4],trimester)
        for min_size in tqdm(range(min_size, 2000, -5000)):
            detection, touches_edge, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192 = find_brain_48(positions_48, min_size, max_size, combined_model_48, img, mask , validation_path,trimester)
            if detection:
                break
            
        
    return detection, touches_edge, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192

def first_stage_gmm_prediction(img, mask, mom, positions_48):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    combined_model_48 = get_model(models[4]) 

    for min_size in tqdm(range(min_size, 5000, -5000)):
        detection, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192 = find_brain_48(positions_48, min_size, max_size, combined_model_48, img, mask)
        if detection:
            break

    return detection, pred_48, valid_position_index_192, cube_48, mask_48, first_pred_192
    
def first_half_stage_prediction(img, mask, mom, positions_36,pred_192_first,trimester):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    combined_model_36 = get_model(models[2],trimester) 

    for min_size in tqdm(range(min_size, 0, -5000)):
        detection, pred_24 ,valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 = find_brain_36(positions_36, min_size, max_size, combined_model_36, img, mask,pred_192_first,trimester)
        print("first_half_stage_prediction",np.sum(first_pred_192))
        if detection:
            break

    return detection, pred_24, valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 > 0 

def first_8th_stage_prediction(img, mask, mom, positions_32,pred_192_first,trimester):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    combined_model_36 = get_model(models[0],trimester) 

    for min_size in tqdm(range(min_size, 0, -5000)):
        detection, pred_24 ,valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 = find_brain_32(positions_32, min_size, max_size, combined_model_36, img, mask,pred_192_first)
        print("first_half_stage_prediction",np.sum(first_pred_192))
        if detection:
            break

    return detection, pred_24, valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 > 0

def first_4th_stage_prediction(img, mask, mom, positions_34,pred_192_first,trimester):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    combined_model_36 = get_model(models[1],trimester) 

    for min_size in tqdm(range(min_size, 0, -5000)):
        detection, pred_24 ,valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192 = find_brain_34(positions_34, min_size, max_size, combined_model_36, img, mask,pred_192_first)
        print("first_half_stage_prediction",np.sum(first_pred_192))
        if detection:
            break

    return detection, pred_24, valid_position_index_48, valid_position_index_192, cube_24, mask_24, first_pred_192
    
def second_stage_prediction(img, mask, pred_48, cube_48, mask_48, valid_position_index_192, mom, positions_24,trimester):
    detection = False
    combined_model_24 = get_model(models[2],trimester) 
    min_size , max_size = get_min_max_size(2,mom)
    if trimester == 2:
        max_size = 5000000
    print("second stage has started.. ")
    for min_size in tqdm(range(min_size, -1000, -5000)):
        print("min_size",min_size)
        detection, pred_24, pred_48, valid_position_index_48,valid_position_index_192, cube_24, mask_24, second_pred_192 = find_brain_24(positions_24, min_size, max_size, combined_model_24, pred_48, cube_48, mask_48, valid_position_index_192, img, mask, trimester)
        if detection:
            break
    if not detection:
        print("not found in second stage .. ")
        raise ValueError("No mask found in this stage!")
    print("second stage has been found.. ")

    return detection, pred_24, pred_48, valid_position_index_48, valid_position_index_192, cube_24, mask_24, second_pred_192 > 0

def second_stage_skip_prediction(img, mask, pred_48, cube_48, mask_48, valid_position_index_192, mom, positions_24,trimester):
    detection = False
    combined_model_24 = get_model(models[2],trimester) 
    min_size , max_size = get_min_max_size(2,mom)
    print("second stage has started.. ")
    for min_size in tqdm(range(min_size, 5000, -5000)):
        detection, pred_48, valid_position_index_48, cube_24, mask_24, second_pred_192 = find_brain_skip_24(positions_24, min_size, max_size, combined_model_24, 
                                                                                                       pred_48, cube_48, mask_48, valid_position_index_192, img, mask)
        if detection:
            break
    if not detection:
        print("not found in second stage .. ")
        raise ValueError("No mask found in this stage!")
    print("second stage has been found.. ")

    return detection, pred_48, valid_position_index_48, cube_24, mask_24, second_pred_192
    
def third_stage_prediction(img, mask, pred_24, cube_24, mask_24, pred_48, cube_48, mask_48, valid_position_index_48, valid_position_index_192, mom, positions_12,trimester):
    detection = False
    combined_model_12 = get_model(models[1],trimester) 
    min_size , max_size = get_min_max_size(3,mom)
    pred_24 = np.zeros_like(mask_24)
    print("3rd stage : min size",min_size)
    for min_size in tqdm(range(min_size, -5000, -5000)):
        detection, pred_24, valid_position_index_24, cube_12, mask_12, third_pred_192 = find_brain_12(positions_12, min_size, max_size,combined_model_12 , pred_24, cube_24,mask_24,pred_48, cube_48,mask_48,img,mask,valid_position_index_48, valid_position_index_192,trimester)
        if detection:
            break
    if not detection:
        raise ValueError("No mask found in this stage!")
    return detection, pred_24, valid_position_index_24, cube_12, mask_12, third_pred_192 > 0

def fourth_stage_prediction(img, mask, pred_12, cube_12, mask_12, cube_24, mask_24, cube_48, mask_48, valid_position_index_24, valid_position_index_48, valid_position_index_192, mom, positions_6,trimester):
    detection = False

    combined_model_6 = get_model(models[0],trimester) 
    min_size , max_size = get_min_max_size(4,mom)
    
    list_pred_192 = []

    for min_size in tqdm(range(min_size, -5000, -5000)):
        detection, pred_6, valid_position_index_12, cube_6, fourth_pred_192 = find_brain_6(positions_6, min_size, max_size, combined_model_6, 
                                                                                            pred_12, cube_12, mask_12, cube_24, mask_24, cube_48, mask_48, img, mask, 
                                                                                            valid_position_index_24, valid_position_index_48, valid_position_index_192)
        if detection:
            break
            
    if not detection:
        raise ValueError("No mask found in this stage!")
    return detection, pred_12, valid_position_index_12, cube_6, fourth_pred_192

def final_step(pred_list, img, mask,trimester):
    try:
        if np.sum(final_mask)==0:
            print("no non zero in final step")
            return np.zeros_like(mask)
        final_mask = pred_list[0]  # np.sum(pred_list, axis=0)
        combined_model_48 = get_model(models[3],trimester) 
        x1, y1, z1, x2, y2, z2 = find_bounding_box(final_mask > 0, cube_size=param_3d.img_size_48)
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_48)
        pred_192 = np.zeros_like(mask)
        prediction_one_hot = combined_model_48.predict(cube_48[None, ..., None], verbose=0)
        prediction = np.argmax(prediction_one_hot, axis=-1)
        prediction[prediction != 0] = 1
        pred_192[x1:x2, y1:y2, z1:z2] = prediction
        return pred_192
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.zeros_like(mask)
    


def get_neighboring_cubes(coords, cube_size, stride, max_dim):
    x1, y1, z1, x2, y2, z2 = coords
    neighbors = []
    
    # Define possible shifts for neighboring cubes (6-directional adjacency)
    shifts = [
        (-stride, 0, 0), (stride, 0, 0), # left, right
        (0, -stride, 0), (0, stride, 0), # down, up
        (0, 0, -stride), (0, 0, stride)  # back, front
    ]
    
    for dx, dy, dz in shifts:
        nx1 = x1 + dx
        ny1 = y1 + dy
        nz1 = z1 + dz
        nx2 = x2 + dx
        ny2 = y2 + dy
        nz2 = z2 + dz
        
        # Ensure the neighboring cube fits within the mask dimensions
        if 0 <= nx1 < max_dim and 0 <= nx2 <= max_dim and \
           0 <= ny1 < max_dim and 0 <= ny2 <= max_dim and \
           0 <= nz1 < max_dim and 0 <= nz2 <= max_dim:
            neighbors.append((nx1, ny1, nz1, nx2, ny2, nz2))
    
    return neighbors
    
def find_brain_48(positions, min_size, max_size,combined_model , img,mask,validation_path,trimester):
    detection = False
    margins_list = [(0, 0, 0)]#, (3, 3, 3),(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]

    denoise_list = [0]
    valid_position_index_192=None
    # orig_vox_size = img.geom.voxsize
    list_pred_192 = []
    pred_48=np.zeros((param_3d.img_size_48,)*3)
    # orig_vox_size = img.geom.voxsize
    print("image.geom.voxsize",img.geom.voxsize)



    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_48)
    x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_48, 192) 



        
    for i in range(len(positions)):
        for margin in margins_list:
            for denoise in denoise_list:
                if denoise==1:
                    image = denoise_img#apply_gaussian_smoothing(img.data[None, ..., None],sigma = 0.9,kernel_size = 2)[0, ..., 0]       
                elif denoise==2:
                    image = resized_big_img
                elif denoise==3:
                    image = resized_small_img
                else:
                    image = img
                pred_192 = np.zeros_like(mask)
                cube_48=np.zeros((param_3d.img_size_48,)*3)
                mask_48=np.zeros((param_3d.img_size_48,)*3)
                
                x1, y1, z1, x2, y2, z2 = positions[i]
                cube = extract_cube(image,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
                x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_48, param_3d.img_size_192)
                cube=cube[None,...,None]
                prediction_one_hot = combined_model.predict(cube, verbose=0)
                prediction = np.argmax(prediction_one_hot,axis=-1)
            
                prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
                
                prediction[prediction != 0] = 1
        
                non_zero_count = np.count_nonzero(prediction)
                if min_size <= non_zero_count:# and is_centered_3d(prediction,margins=margin):
                    pred_192[x1:x2, y1:y2, z1:z2] = prediction
                    
                    list_pred_192.append(pred_192)
    
                    detection = True



    print("detection",detection)


    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask
        cutoff = len(unique_vals) * 2 // 5  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)


    touches_edge = 0
    
    if detection:
        pred_192 = combine_masks_sum(list_pred_192)
        pred_192_temp =adjust_weights(pred_192)
        threshold = np.max(pred_192_temp)-1
        pred_192 = pred_192_temp > threshold
        pred_192 = find_large_components(pred_192,min_area=2000, max_area=param_3d.max_area)
            
            
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192,cube_size=param_3d.img_size_48)
        valid_position_index_192 =(x1, y1, z1, x2, y2, z2)

        pred_48 = extract_cube(pred_192,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )

            
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
        mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        pred_48 = extract_cube(pred_192,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        
        ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)   
        ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
        ne.plot.volume3D(cube_48,slice_nos=ms);
        ne.plot.volume3D(pred_48,slice_nos=ms);
        
        # pred_192 = adjust_weights(pred_192,trimester)
        print("### 48Net Hard dice: ", my_hard_dice(pred_192, mask.data))
    return detection , touches_edge, pred_48, valid_position_index_192, cube_48,mask_48, pred_192
    



def find_brain_36(positions, min_size, max_size, combined_model, img, mask, pred_192_first,trimester):
    detection = False
    margins_list = [(0, 0, 0)]#, (2, 2, 2), (3, 3, 3),(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]
    denoise_list = [0]
    valid_position_index_192=None

    list_pred_192 = []
    pred_48=np.zeros((param_3d.img_size_24,)*3)
    pred_24 = np.zeros((param_3d.img_size_24,) * 3)
    cube_24 = np.zeros((param_3d.img_size_24,) * 3)
    mask_24 = np.zeros((param_3d.img_size_24,) * 3)


    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_24)
    x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_24, 192) 

        
    for i in range(len(positions)):
        for margin in margins_list:
            for denoise in denoise_list:
                if denoise:
                    image = denoise_img
                else:
                    image = img
                pred_192 = np.zeros_like(mask)
                cube_48=np.zeros((param_3d.img_size_24,)*3)
                mask_48=np.zeros((param_3d.img_size_24,)*3)
                


                x1, y1, z1, x2, y2, z2 = positions[i]
                cube = extract_cube(image,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24 )
                cube=cube[None,...,None]
                prediction_one_hot = combined_model.predict(cube, verbose=0)
                prediction = np.argmax(prediction_one_hot,axis=-1)

                # prediction = prediction[0]
                prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
                # prediction = find_large_components(prediction,min_area=param_3d.min_area, max_area=param_3d.max_area)
                
                prediction[prediction != 0] = 1
                # valid_position_index_192 = (x1, y1, z1, x2, y2, z2)

                
        
                non_zero_count = np.count_nonzero(prediction)
    
                if min_size <= non_zero_count:# and is_centered_3d(prediction,margins=margin):
                    valid_position_index_192 = x1, y1, z1, x2, y2, z2 #= find_bounding_box(pred_48>0, cube_size=param_3d.img_size_24)                
                    pred_192[x1:x2, y1:y2, z1:z2] = prediction
                    list_pred_192.append(pred_192)
    
                    detection = True
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_24)
    valid_position_index_48 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_48)
    valid_position_index_192 = x1, y1, z1, x2, y2, z2
    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask
        if trimester ==3:
            cutoff = len(unique_vals) * 2 // 5
        else:
            cutoff = len(unique_vals) * 1 // 5  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)
        
    if detection:
        pred_192_temp = combine_masks_sum(list_pred_192)
        pred_192_temp = adjust_weights(pred_192_temp)
        threshold = np.max(pred_192_temp)-1
        pred_192 = pred_192_temp>threshold 
        # pred_192 = find_large_components(pred_192,min_area=param_3d.min_area, max_area=param_3d.max_area)
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192 , cube_size=param_3d.img_size_48)
        x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_48, param_3d.img_size_192) 

        valid_position_index_192 =(x1, y1, z1, x2, y2, z2)
        # valid_position_index_48 = (x1, y1, z1, x2, y2, z2)
        print("valid_position_index_192 from 48",x1, y1, z1, x2, y2, z2,x2-x1,y2-y1,z2-z1)
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
        mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        pred_48 = extract_cube(pred_192,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )

        # pred_48 = adjust_weights(pred_48)

        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48,cube_size=param_3d.img_size_24)
        cube_24 = extract_cube(cube_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        mask_24 = extract_cube(mask_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        valid_position_index_48 = x1, y1, z1, x2, y2, z2

        ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)   
        ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
        ne.plot.volume3D(cube_48,slice_nos=ms);
        ne.plot.volume3D(pred_48,slice_nos=ms);
        
        print("### 36Net Hard dice: ", my_hard_dice(pred_192, mask.data))
    return detection, pred_24,valid_position_index_48, valid_position_index_192, cube_24, mask_24, pred_192


def find_brain_34(positions, min_size, max_size, combined_model, img, mask, pred_192_first):
    detection = False
    margins_list = [(1,1,1)]#, (2, 2, 2), (3, 3, 3),(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]
    denoise_list = [0]
    valid_position_index_192=None

    list_pred_192 = []
    pred_48=np.zeros((param_3d.img_size_12,)*3)
    pred_24 = np.zeros((param_3d.img_size_12,) * 3)
    cube_24 = np.zeros((param_3d.img_size_12,) * 3)
    mask_24 = np.zeros((param_3d.img_size_12,) * 3)


    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_12)
        
    for i in range(len(positions)):
        for margin in margins_list:
            for denoise in denoise_list:

                image = img
                pred_192 = np.zeros_like(mask)
                cube_48=np.zeros((param_3d.img_size_12,)*3)
                mask_48=np.zeros((param_3d.img_size_12,)*3)
                


                x1, y1, z1, x2, y2, z2 = positions[i]
                cube = extract_cube(image,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_12 )
                cube=cube[None,...,None]
                prediction_one_hot = combined_model.predict(cube, verbose=0)
                prediction = np.argmax(prediction_one_hot,axis=-1)
            
                prediction = prediction[0]#ndimage.binary_fill_holes(prediction[0]).astype(int)
                prediction = find_large_components(prediction,min_area=param_3d.min_area, max_area=param_3d.max_area)
                
                
                non_zero_count = np.count_nonzero(prediction)
    
                if min_size <= non_zero_count:# and is_centered_3d(prediction,margins=margin):
                    valid_position_index_192 = x1, y1, z1, x2, y2, z2 #= find_bounding_box(pred_48>0, cube_size=param_3d.img_size_24)      
                    pred_192[x1:x2, y1:y2, z1:z2] = prediction
                    list_pred_192.append(pred_192)
    
                    detection = True

    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_12)
    valid_position_index_24 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_24)
    valid_position_index_48 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_48)
    valid_position_index_192 = x1, y1, z1, x2, y2, z2
    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask
    
        cutoff = len(unique_vals) * 1 // 10  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)
        
    if detection:
        pred_192_temp = combine_masks_sum(list_pred_192)
        pred_192_temp = adjust_weights(pred_192_temp)
        threshold = np.max(pred_192_temp)/2
        pred_192 =  pred_192_temp > threshold
        # pred_192 = find_largest_component(pred_192)
        # pred_192 = find_large_components(pred_192,min_area=param_3d.min_area, max_area=param_3d.max_area)
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192 , cube_size=param_3d.img_size_48)
        valid_position_index_192 =(x1, y1, z1, x2, y2, z2)
        # valid_position_index_48 = (x1, y1, z1, x2, y2, z2)
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
        mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        pred_48 = extract_cube(pred_192_temp,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        # pred_48 = adjust_weights(pred_48)

        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48,cube_size=param_3d.img_size_24)
        cube_24 = extract_cube(cube_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        mask_24 = extract_cube(mask_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        valid_position_index_48 = x1, y1, z1, x2, y2, z2
        ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)   
        ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
        ne.plot.volume3D(cube_48,slice_nos=ms);
        ne.plot.volume3D(pred_48,slice_nos=ms);
        
        print("### 34Net Hard dice: ", my_hard_dice(pred_192, mask.data))
    return detection, pred_24,valid_position_index_48, valid_position_index_192, cube_24, mask_24, pred_192


def find_brain_32(positions, min_size, max_size, combined_model, img, mask, pred_192_first):
    detection = False
    margins_list = [(1, 0 ,0, 0)]#, (2, 2, 2), (3, 3, 3),(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]
    denoise_list = [0]
    valid_position_index_192=None

    list_pred_192 = []
    pred_48=np.zeros((param_3d.img_size_24,)*3)
    pred_24 = np.zeros((param_3d.img_size_24,) * 3)
    cube_24 = np.zeros((param_3d.img_size_24,) * 3)
    mask_24 = np.zeros((param_3d.img_size_24,) * 3)

    # noise_model = get_model("noise")
    # denoise_img = noise_model.predict(img.data[None,...,None], verbose=0)
    # denoise_img = denoise_img[0,...,0]

    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_6)
        
    for i in range(len(positions)):
        for margin in margins_list:
            for denoise in denoise_list:
                if denoise:
                    image = denoise_img
                else:
                    image = img
                pred_192 = np.zeros_like(mask)
                cube_48=np.zeros((param_3d.img_size_6,)*3)
                mask_48=np.zeros((param_3d.img_size_6,)*3)
                


                x1, y1, z1, x2, y2, z2 = positions[i]
                cube = extract_cube(image,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_6 )
                cube=cube[None,...,None]
                prediction_one_hot = combined_model.predict(cube, verbose=0)
                prediction = np.argmax(prediction_one_hot,axis=-1)
            
                prediction = prediction[0]#ndimage.binary_fill_holes(prediction[0]).astype(int)
                # prediction = find_large_components(prediction,min_area=param_3d.min_area, max_area=param_3d.max_area)
                
                prediction[prediction != 0] = 1
                # valid_position_index_192 = (x1, y1, z1, x2, y2, z2)

                
        
                non_zero_count = np.count_nonzero(prediction)
    
                if min_size <= non_zero_count:# and is_centered_in_plane(prediction, margins=margin):
                    valid_position_index_192 = x1, y1, z1, x2, y2, z2 #= find_bounding_box(pred_48>0, cube_size=param_3d.img_size_24)                
                    pred_192[x1:x2, y1:y2, z1:z2] = prediction
                    list_pred_192.append(pred_192)
    
                    detection = True
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_6)
    valid_position_index_12 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_12)
    valid_position_index_24 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_24)
    valid_position_index_48 = x1, y1, z1, x2, y2, z2
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_48)
    valid_position_index_192 = x1, y1, z1, x2, y2, z2
    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask

        cutoff = len(unique_vals) * 1 // 10  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)
        
    if detection:
        pred_192_temp = combine_masks_sum(list_pred_192)
        pred_192_temp = adjust_weights(pred_192_temp)
        threshold = np.max(pred_192_temp)/2
        pred_192 =  pred_192_temp > threshold
        # pred_192 = find_large_components(pred_192,min_area=param_3d.min_area, max_area=param_3d.max_area)
        # pred_192 = combine_masks_majority_voting([pred_192_temp>0,pred_192_first>0])
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192 , cube_size=param_3d.img_size_48)
        valid_position_index_192 =(x1, y1, z1, x2, y2, z2)
        # valid_position_index_48 = (x1, y1, z1, x2, y2, z2)
        print("valid_position_index_192 from 48",x1, y1, z1, x2, y2, z2,x2-x1,y2-y1,z2-z1)
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
        mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        pred_48 = extract_cube(pred_192_temp,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )

        # pred_48 = adjust_weights(pred_48)

        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48,cube_size=param_3d.img_size_24)
        cube_24 = extract_cube(cube_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        mask_24 = extract_cube(mask_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        valid_position_index_48 = x1, y1, z1, x2, y2, z2

        ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)   
        ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
        ne.plot.volume3D(cube_48,slice_nos=ms);
        ne.plot.volume3D(pred_48,slice_nos=ms);
        
        print("### 32Net Hard dice: ", my_hard_dice(pred_192, mask.data))
    return detection, pred_24,valid_position_index_48, valid_position_index_192, cube_24, mask_24, pred_192



def find_brain_24(positions, min_size, max_size, combined_model, pred_48, img_48, mask_48, valid_position_index_192, img, mask, trimester):
    detection = False
    list_pred_192 = []
    valid_position_index_48 = None
    pred_192 = np.zeros_like(mask)
    margins_list = [
        (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0),
        (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 1, 0), (1, 1, 0, 1),
        (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1), (2, 2, 2, 2)
    ]

    if trimester == 2:
        margins_list = [(0, 0, 0)]
    else:
        margins_list = [(2, 2, 2)]

    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_24)
    x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_24, param_3d.img_size_48)

        
    denoise_list = [0]
    x1, y1, z1, x2, y2, z2 = valid_position_index_192

    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask_48,cube_size=param_3d.img_size_24)
    
    neighs = get_neighboring_cubes((x1, y1, z1, x2, y2, z2 ),param_3d.img_size_24,4,128)
    for a in neighs:
        positions.append(a)


    print("24 started", valid_position_index_192)

    for i in range(len(positions)):
        # try:
        for margin in margins_list:
            # for denoise in denoise_list:
            image = img_48

            x1, y1, z1, x2, y2, z2 = positions[i]
            pred_48 = np.zeros((param_3d.img_size_48,) * 3)
            pred_192 = np.zeros_like(mask)
            cube_24 = np.zeros((param_3d.img_size_24,) * 3)
            mask_24 = np.zeros((param_3d.img_size_24,) * 3)

            cube = extract_cube(image, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
            cube = cube[None, ..., None]
            prediction_one_hot = combined_model.predict(cube, verbose=0)
            prediction = np.argmax(prediction_one_hot, axis=-1)[0]
            prediction[prediction != 0] = 1
            non_zero_count = np.count_nonzero(prediction)
            if 0 <= non_zero_count:# and is_centered_3d(prediction, margins=margin):
                pred_48[x1:x2, y1:y2, z1:z2] = prediction
                x1, y1, z1, x2, y2, z2 = valid_position_index_192
                pred_192[x1:x2, y1:y2, z1:z2] = pred_48

                list_pred_192.append(pred_192)
                detection = True

    print("find brain 24 detection", detection, len(list_pred_192))

    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask
        if trimester ==3:
            cutoff = len(unique_vals) * 2 // 5  # 40% cutoff
        else:
            cutoff = len(unique_vals) * 1 // 5  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)

    
    
    pred_24 = np.zeros((param_3d.img_size_24,) * 3)
    cube = np.zeros((param_3d.img_size_24,) * 3)
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=param_3d.img_size_24)
    valid_position_index_48 = x1, y1, z1, x2, y2, z2

    if detection:
        try:
            pred_192 = combine_masks_sum(list_pred_192)
            threshold = np.max(pred_192)-1
            pred_192_temp = adjust_weights(pred_192)
            
            pred_192 = pred_192_temp> threshold
            # pred_192=find_largest_component(pred_192)
            # pred_192 = find_large_components(pred_192,min_area=2000, max_area=param_3d.max_area)
            x1, y1, z1, x2, y2, z2 = valid_position_index_192
            pred_48 = extract_cube(pred_192, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_48)

            
            x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48 , cube_size=param_3d.img_size_24)
            x1, y1, z1, x2, y2, z2 = calibrate_positions(x1, y1, z1, x2, y2, z2, param_3d.img_size_24, param_3d.img_size_48)
            valid_position_index_48 = (x1, y1, z1, x2, y2, z2)

            pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
            
            cube_24 = extract_cube(img_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
            mask_24 = extract_cube(mask_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
            pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
            
            ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)
            
            
            ms = np.mean(np.column_stack(np.nonzero(mask_24)), axis=0).astype(int)

            ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
            
            ne.plot.volume3D(cube_24,slice_nos=ms)
            ne.plot.volume3D(pred_24,slice_nos=ms)
            # pred_192 = adjust_weights(pred_192)
            # pred_192 = find_largest_component(pred_192_temp>threshold)
            pred_192 = pred_192_temp
            print("### 24Net Hard dice: ", my_hard_dice(pred_192>0 , mask.data))
        except Exception as e:
            print(f"Error in final processing: {e}")
            
    
    return detection, pred_24, pred_48, valid_position_index_48, valid_position_index_192, cube_24, mask_24, pred_192

def find_brain_skip_24(positions, min_size, max_size,combined_model , pred_48, img_48,mask_48,valid_position_index_192, img,mask ):
    detection = False

    list_pred_192 = []
    valid_position_index_48 = None
    margins_list = [(1, 0, 0, 0),(0, 1, 0, 0),(0, 0, 1, 0),(0, 0, 0, 1),(1, 1, 0, 0), (1, 0, 1, 0),(1, 0, 0, 1),(0, 1, 1, 0),(0, 1, 0, 1),(0, 0, 1, 1),(1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1),(1, 1, 1, 1),(2, 2, 2, 2)]
    pred_192 = np.zeros_like(mask)
    denoise_list = [0,1]
    detection = True
    pred_192 = np.zeros_like(mask)
    x1, y1, z1, x2, y2, z2 = valid_position_index_192
    pred_192[x1:x2, y1:y2, z1:z2]=pred_48 
    
    print("find brain 24 detection",detection,len(list_pred_192))

    x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48>0,cube_size=param_3d.img_size_24)
    
    cube_24 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_24 )
    mask_24 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24 )
    pred_24 = extract_cube(pred_48,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24 )

    x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48>0,cube_size=param_3d.img_size_24)
    valid_position_index_48 =(x1, y1, z1, x2, y2, z2)

    ms = np.mean(np.column_stack(np.nonzero(mask_24)), axis=0).astype(int)    
    ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
    ne.plot.volume3D(cube_24,slice_nos=ms);
    ne.plot.volume3D(pred_24,slice_nos=ms);
    
    pred_192 = pred_192>0
    print("### 24Net Hard dice: ", my_hard_dice(pred_192, mask.data))
    return detection, pred_48 , valid_position_index_48, cube_24 ,mask_24,  pred_192
    

def find_brain_12(positions, min_size, max_size,combined_model , pred_24, img_24,mask_24,pred_48, img_48,mask_48,img,mask,valid_position_index_48, valid_position_index_192,trimester):
    detection = False
    margins_list = [(0, 0, 0)]#,(3, 3, 3),(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]
    if trimester ==2:
        margins_list = [(0, 0, 0)]
    denoise_list = [1]
    list_pred_192 = []
    print("12Net has started .. ")
    valid_position_index_24 = None

            
    # try:
    for i in range(len(positions)):
        for margin in margins_list:

            image = img_24
                
            x1, y1, z1, x2, y2, z2 = positions[i]

            pred_24 = np.zeros_like(mask_24)
            pred_48 = np.zeros_like(mask_48)
            pred_192 = np.zeros_like(mask)
            
            mask_12 = extract_cube(mask_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
            cube = extract_cube(image, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
            cube = cube[None, ..., None]
            
            prediction_one_hot = combined_model.predict(cube, verbose=0)
            prediction = np.argmax(prediction_one_hot, axis=-1)[0]
            prediction[prediction != 0] = 1
            non_zero_count = np.count_nonzero(prediction)
            prediction = find_large_components(prediction,min_area=param_3d.min_area, max_area=param_3d.max_area)
            if min_size <= non_zero_count:# and is_centered_3d(prediction, margins=margin):
                pred_24[x1:x2, y1:y2, z1:z2] = prediction
                x1, y1, z1, x2, y2, z2 = valid_position_index_48
                pred_48[x1:x2, y1:y2, z1:z2] = pred_24
                # print("ggg",valid_position_index_192,pred_192.shape)
                x1, y1, z1, x2, y2, z2 = valid_position_index_192
                # print("ddd",x2-x1,y2-y1,z2-z1,pred_24.shape,pred_48.shape,pred_192.shape)
                pred_192[x1:x2, y1:y2, z1:z2] = pred_48
                list_pred_192.append(pred_192)
                detection = True

    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            print("*** only one unique ***")
            return pred_mask
        if trimester ==3:
            cutoff = len(unique_vals) * 3 // 5  # 40% cutoff
        else:
            cutoff = len(unique_vals) * 1 // 5  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)


    print("find brain 12 detection",detection)
    pred_12 = np.zeros((param_3d.img_size_12,) * 3)
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask_24 > 0, cube_size=param_3d.img_size_12)
    mask_12 = extract_cube(mask_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
    cube_12 = extract_cube(img_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
    valid_position_index_24 = x1, y1, z1, x2, y2, z2
    if detection:
        pred_192 = combine_masks_sum(list_pred_192)
        threshold = np.max(pred_192)-1
        pred_192 = adjust_weights(pred_192)
        pred_192 = pred_192>threshold #>0#adjust_weights(pred_192)
        # pred_192 = find_largest_component(pred_192)
        
        x1, y1, z1, x2, y2, z2 = valid_position_index_192
        pred_48 = extract_cube(pred_192, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_48)
        
        x1, y1, z1, x2, y2, z2 = valid_position_index_48
        pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_24, cube_size=param_3d.img_size_12)
        valid_position_index_24 = (x1, y1, z1, x2, y2, z2)
        mask_12 = extract_cube(mask_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
        cube_12 = extract_cube(img_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
    
        pred_12 = extract_cube(pred_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
        
        # x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_24>threshold,cube_size=param_3d.img_size_12)
        # valid_position_index_24 =(x1, y1, z1, x2, y2, z2)
        ms = np.mean(np.column_stack(np.nonzero(pred_12)), axis=0).astype(int)
        ms = (0, 0, 0) if any(x < 0 or x > 128 for x in ms) else ms
        ne.plot.volume3D(cube_12,slice_nos=ms);
        ne.plot.volume3D(pred_12,slice_nos=ms);
        pred_192 = pred_192> 0
        print("### 12Net Hard dice: ", my_hard_dice(pred_192>0, mask.data))
    return detection, pred_12 , valid_position_index_24, cube_12 , mask_12, pred_192



def find_brain_6(positions, min_size, max_size, combined_model, pred_12, img_12, mask_12, img_24, mask_24, img_48, mask_48, img, mask, valid_position_index_24, valid_position_index_48, valid_position_index_192):
    detection = False
    print("6Net has started .. ")
    
    list_pred_192 = []
    denoise_list = [1]
    valid_position_index_12 = None
    num_detections = 0
    margins_list = [ (1, 0, 0, 0),(0, 0, 0, 1),
        (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1), (2, 2, 2, 2)
    ]
    margins_list = [(0, 0, 0)]#,(2, 2, 2)]#,(1, 2, 1),(2, 1, 1)]#,(5, 5, 5),(8, 8, 8),(24, 24, 24),(32, 32, 32)]
    
    
    pred_6 = np.zeros((param_3d.img_size_6,) * 3)
    for i in range(len(positions)):
        # try:
        for margin in margins_list:
            for denoise in denoise_list:

                image = img_24
                    
                pred_192 = np.zeros_like(mask)
                pred_12 = np.zeros_like(mask_12)
                pred_24 = np.zeros_like(mask_24)
                pred_48 = np.zeros_like(mask_48)
                cube_6 = np.zeros((param_3d.img_size_6,) * 3)
                
                x1, y1, z1, x2, y2, z2 = positions[i]
                cube = extract_cube(image, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_6)
                cube = cube[None, ..., None]
                cube_6 = extract_cube(image, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_6)
                prediction_one_hot = combined_model.predict(cube, verbose=0)
                prediction = np.argmax(prediction_one_hot, axis=-1)[0]
                prediction = ndimage.binary_fill_holes(prediction).astype(int)

                
                # prediction = prediction[0]
                prediction[prediction != 0] = 1
                prediction = find_largest_component(prediction)
                non_zero_count = np.count_nonzero(prediction)
                if min_size <= non_zero_count:# and is_centered_3d(prediction, margins=margin):
                    pred_24[x1:x2, y1:y2, z1:z2] = prediction
                    x1, y1, z1, x2, y2, z2 = valid_position_index_48
                    pred_48[x1:x2, y1:y2, z1:z2] = pred_24
                    x1, y1, z1, x2, y2, z2 = valid_position_index_192
                    pred_192[x1:x2, y1:y2, z1:z2] = pred_48
                    list_pred_192.append(pred_192)
                    detection = True
    print("find brain 6Net detection",detection,len(list_pred_192))
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask_12 > 0, cube_size=param_3d.img_size_6)
    mask_6 = extract_cube(mask_12, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_6)
    
    def adjust_weights(pred_mask):
        flat_mask = pred_mask.flatten()
        unique_vals = np.unique(flat_mask[flat_mask > 0])
        if len(unique_vals) <= 1:
            return pred_mask
        cutoff = len(unique_vals) * 1 // 5  # 40% cutoff
        least_40_percent = unique_vals[:cutoff]
        max_weight = np.max(flat_mask)
        return np.where(np.isin(flat_mask, least_40_percent) | (flat_mask == 0), flat_mask, max_weight).reshape(pred_mask.shape)
            
    if detection:
        # try:
        pred_192 = combine_masks_sum(list_pred_192)
        pred_192 = adjust_weights(pred_192)
        threshold = np.max(pred_192)-1#/2
        pred_192 = pred_192 > threshold
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192>np.max(pred_192)-1 , cube_size=param_3d.img_size_48)
        valid_position_index_192 =(x1, y1, z1, x2, y2, z2)
        # valid_position_index_48 = (x1, y1, z1, x2, y2, z2)
        print("valid_position_index_192 from 48",x1, y1, z1, x2, y2, z2,x2-x1,y2-y1,z2-z1)
        cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
        mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        pred_48 = extract_cube(pred_192,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        
        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_48>threshold,cube_size=param_3d.img_size_24)
        cube_24 = extract_cube(cube_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        mask_24 = extract_cube(mask_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        pred_24 = extract_cube(pred_48, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_24)
        valid_position_index_48 = x1, y1, z1, x2, y2, z2

        x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_24>threshold,cube_size=param_3d.img_size_12)
        cube_12 = extract_cube(cube_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
        mask_12 = extract_cube(mask_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
        pred_12 = extract_cube(pred_24, x1, y1, z1, x2, y2, z2, cube_size=param_3d.img_size_12)
        valid_position_index_24 = x1, y1, z1, x2, y2, z2
        
        ne.plot.volume3D(cube_12);
        ne.plot.volume3D(pred_12);

        pred_192 = pred_192>0
        print("### 6Net Hard dice: ", my_hard_dice(pred_192 , mask.data))


    return detection, pred_6, valid_position_index_12, cube_6, pred_192
