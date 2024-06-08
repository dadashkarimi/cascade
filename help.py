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
    
        en = [32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,2]
        
        input_img = Input(shape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1))
        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1), nb_features=(en, de), batch_norm=False,
                           nb_conv_per_level=2,
                           final_activation_function='softmax')
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=param_3d.img_size_24)
        
        if random:
            x1, y1, z1, x2, y2, z2 = find_random_bounding_box(mask,cube_size=param_3d.img_size_24,margin=param_3d.img_size_12,full_random=full_random)
    
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
    
        en = [32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,2]
        
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