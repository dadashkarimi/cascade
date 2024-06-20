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
def get_cube_and_model(model,img, mask, random,full_random):
    epsilon =1e-7
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
            
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_6net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)
    
        
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2)
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
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_12net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_12)
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
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_24net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24)
        
    elif model=="48Net":
        print("48Net model is loading")
        
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_48)
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_48,full_random=full_random)
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_48net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
        # input_img =  apply_gaussian_smoothing(input_img,sigma = 1.0,kernel_size = 3)
        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)    
        cube = extract_cube(img.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48)
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
    box = np.zeros((192, 192, 192), dtype=int)
    box[x1:x2+1, y1:y2+1, z1:z2+1] = 1
    return cube , box, combined_model

def get_model(model):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    if model=="6Net":
        print("6Net model is loading")
        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        input_img = Input(shape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
            
        latest_weight = max(glob.glob(os.path.join("models_cascade_6net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
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
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_12net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
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
        latest_weight = max(glob.glob(os.path.join("models_cascade_24net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

        generated_img_norm = min_max_norm(input_img)
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.load_weights(latest_weight)            
    elif model=="48Net":
        print("48Net model is loading")
        
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]
        
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
    
        latest_weight = max(glob.glob(os.path.join("models_cascade_48net", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
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
    return combined_model

def is_centered_in_plane(prediction, margins=(8, 8, 8, 8)):
    summed_projection_axis2 = np.sum(prediction, axis=2)
    
    summed_projection_axis0 = np.sum(summed_projection_axis2, axis=0)
    summed_projection_axis1 = np.sum(summed_projection_axis2, axis=1)
    
    non_zero_coords_axis0 = np.argwhere(summed_projection_axis0)
    non_zero_coords_axis1 = np.argwhere(summed_projection_axis1)
    
    min_coords_axis0 = np.min(non_zero_coords_axis0, axis=0)
    max_coords_axis0 = np.max(non_zero_coords_axis0, axis=0)
    
    min_coords_axis1 = np.min(non_zero_coords_axis1, axis=0)
    max_coords_axis1 = np.max(non_zero_coords_axis1, axis=0)
    
    axis0_centered = np.all(min_coords_axis0 >= margins[0]) and np.all(summed_projection_axis0.shape - max_coords_axis0 - 1 >= margins[1])
    axis1_centered = np.all(min_coords_axis1 >= margins[2]) and np.all(summed_projection_axis1.shape - max_coords_axis1 - 1 >= margins[3])
    
    return axis0_centered and axis1_centered
    
def find_brain_48(positions, min_size, max_size,combined_model , img,mask):
    detection = False
    pred_192 = np.zeros_like(mask)
    valid_position_index_192=None
    cube_48=np.zeros((param_3d.img_size_48,)*3)
    mask_48=np.zeros((param_3d.img_size_48,)*3)

    for i in range(len(positions)):
        
        x1, y1, z1, x2, y2, z2 = positions[i]
        cube = extract_cube(img,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
        cube=cube[None,...,None]
        prediction_one_hot = combined_model.predict(cube, verbose=0)
        prediction = np.argmax(prediction_one_hot,axis=-1)
    
        prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
        prediction = find_largest_component(prediction)
        
        prediction[prediction != 0] = 1

        # prediction = find_largest_component(prediction[0])
        non_zero_count = np.count_nonzero(prediction)
        if  min_size <= non_zero_count and is_centered_in_plane(prediction,margins=(5, 10, 7, 7)):
            print(non_zero_count)
            print(x2-x1,y2-y1,z2-z1)
            cube_48 = extract_cube(img.data, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_48 )
            mask_48 = extract_cube(mask.data,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_48 )
            valid_position_index_192 = (x1, y1, z1, x2, y2, z2)
    
            ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)
            ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
            # ne.plot.volume3D(img,slice_nos=ms);
            # ne.plot.volume3D(mask,slice_nos=ms);
            
            pred_192[x1:x2, y1:y2, z1:z2] = prediction
            ne.plot.volume3D(pred_192,slice_nos=ms);
    
            # summed_projection_axis2 = np.sum(prediction, axis=2)
            print("### 48Net Hard dice: ", my_hard_dice(pred_192, mask.data))

            detection = True
            break
    return detection , valid_position_index_192, cube_48,mask_48, pred_192
    
def find_brain_24(positions, min_size, max_size,combined_model , img_48,mask_48,valid_position_index_192, img,mask):
    detection = False
    pred_48 = np.zeros_like(mask_48)
    pred_192 = np.zeros_like(mask)
    cube_24 = np.zeros((param_3d.img_size_24,)*3)

    valid_position_index_48 = None
    for i in range(len(positions)):
        x1, y1, z1, x2, y2, z2 = positions[i]
        cube = extract_cube(img_48,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24 )
        cube=cube[None,...,None]
        cube_24 = extract_cube(img_48, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_24 )
        mask_24 = extract_cube(mask_48,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_24 )
    
        prediction_one_hot = combined_model.predict(cube, verbose=0)
        prediction = np.argmax(prediction_one_hot,axis=-1)
    
        prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
        prediction[prediction != 0] = 1
        prediction = find_largest_component(prediction)


        non_zero_count = np.count_nonzero(prediction)
        if  min_size <= non_zero_count and is_centered_in_plane(prediction,margins=(10, 6, 10, 10)):
            print(non_zero_count)

            valid_position_index_48 = (x1, y1, z1, x2, y2, z2)

            ms = np.mean(np.column_stack(np.nonzero(mask_48)), axis=0).astype(int)
            
            pred_48[x1:x2, y1:y2, z1:z2] = prediction
            x1, y1, z1, x2, y2, z2 = valid_position_index_192
            pred_192[x1:x2, y1:y2, z1:z2]=pred_48
            ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)

            # ne.plot.volume3D(img,slice_nos=ms);
            ne.plot.volume3D(pred_192,slice_nos=ms);
    
            detection = True
            break
    return detection, pred_48 , valid_position_index_48, cube_24 ,mask_24,  pred_192

def find_brain_12(positions, min_size, max_size,combined_model , img_24,mask_24,img_48,mask_48,img,mask,valid_position_index_192, valid_position_index_48):
    detection = False
    pred_24 = np.zeros_like(mask_24)
    pred_48 = np.zeros_like(mask_48)
    pred_192 = np.zeros_like(mask)

    cube_12 = np.zeros((param_3d.img_size_12,)*3)

    valid_position_index_24 = None
    for i in range(len(positions)):
        x1, y1, z1, x2, y2, z2 = positions[i]
        mask_12 = extract_cube(mask_24,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_12 )
        cube = extract_cube(img_24,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_12 )
        cube=cube[None,...,None]
        prediction_one_hot = combined_model.predict(cube, verbose=0)
        prediction = np.argmax(prediction_one_hot,axis=-1)
    
        prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
        prediction[prediction != 0] = 1
        prediction = find_largest_component(prediction)

        non_zero_count = np.count_nonzero(prediction)
        if  min_size <= non_zero_count and is_centered_in_plane(prediction,margins=(2, 2, 2, 2)):
            print(non_zero_count)
            cube_12 = extract_cube(img_24, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_12 )
            
            valid_position_index_24 = (x1, y1, z1, x2, y2, z2)

            ms = np.mean(np.column_stack(np.nonzero(mask_24)), axis=0).astype(int)
            
            pred_24[x1:x2, y1:y2, z1:z2] = prediction
            x1, y1, z1, x2, y2, z2 = valid_position_index_48
            pred_48[x1:x2, y1:y2, z1:z2]=pred_24
            x1, y1, z1, x2, y2, z2 = valid_position_index_192
            pred_192[x1:x2, y1:y2, z1:z2]=pred_48

            
            ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
            ne.plot.volume3D(pred_192,slice_nos=ms);
    
            detection = True
            break
    return detection, pred_24 , valid_position_index_24, cube_12 , mask_12, pred_192
    
def find_brain_6(positions, min_size, max_size,combined_model , img_12,mask_12,img_24,mask_24,img_48,mask_48,img,mask, valid_position_index_24, valid_position_index_48,valid_position_index_192):
    detection = False

    
    list_pred_192 = []


    valid_position_index_12 = None
    num_detections = 0
    for i in range(len(positions)):
        pred_192 = np.zeros_like(mask)
        pred_12 = np.zeros_like(mask_12)
        pred_24 = np.zeros_like(mask_24)
        pred_48 = np.zeros_like(mask_48)
        cube_6 = np.zeros((param_3d.img_size_6,)*3)
        
        x1, y1, z1, x2, y2, z2 = positions[i]
        cube = extract_cube(img_12,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_6 )
        cube=cube[None,...,None]
        cube_6 = extract_cube(img_12, x1, y1, z1, x2, y2, z2 ,cube_size=param_3d.img_size_6 )
        mask_6 = extract_cube(mask_12,x1, y1, z1, x2, y2, z2,cube_size=param_3d.img_size_6 )
    
        prediction_one_hot = combined_model.predict(cube, verbose=0)
        prediction = np.argmax(prediction_one_hot,axis=-1)
    
        prediction = ndimage.binary_fill_holes(prediction[0]).astype(int)
        prediction[prediction != 0] = 1
        prediction = find_largest_component(prediction)

        non_zero_count = np.count_nonzero(prediction)
        if  min_size <= non_zero_count and is_centered_in_plane(prediction,margins=(1, 1, 1, 1)):
            valid_position_index_12 = (x1, y1, z1, x2, y2, z2)

            ms = np.mean(np.column_stack(np.nonzero(mask_12)), axis=0).astype(int)
            
            pred_12[x1:x2, y1:y2, z1:z2] = prediction
            x1, y1, z1, x2, y2, z2 = valid_position_index_24
            pred_24[x1:x2, y1:y2, z1:z2]=pred_12
            
            x1, y1, z1, x2, y2, z2 = valid_position_index_48
            pred_48[x1:x2, y1:y2, z1:z2]=pred_24

            x1, y1, z1, x2, y2, z2 = valid_position_index_192
            pred_192[x1:x2, y1:y2, z1:z2]=pred_48
            list_pred_192.append(pred_192)

            num_detections =num_detections + 1
            print(non_zero_count)
            # if num_detections ==2:
            #     break
    if num_detections>0:
        pred_192 = combine_masks_union(list_pred_192)
        ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
        ne.plot.volume3D(pred_192,slice_nos=ms);
        detection = True
    # break
    return detection, pred_12 , valid_position_index_12, cube_6 , pred_192
