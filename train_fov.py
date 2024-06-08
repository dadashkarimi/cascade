import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from neurite.tf import models  # Assuming the module's location
import voxelmorph.tf.losses as vtml
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.losses import MeanSquaredError

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
import sys
import nibabel as nib
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full

import tensorflow.keras.layers as KL
import voxelmorph as vxm
from utils import *
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
import os, shutil, glob

from tensorflow.keras.layers import Input, BatchNormalization

# from tensorflow.keras.layers import Input, InstanceNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.0001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.2, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=0.2,help="scale")
parser.add_argument('-body_scale','--body_scale',type=float,default=2,help="body scale")
parser.add_argument('-wm','--warp_max',type=float,default=0.2,help="scale")
parser.add_argument('-bsh','--brain_shift',type=int,default=5,help="brain shift")
parser.add_argument('-body_shift','--body_shift',type=int,default=0,help="brain shift")

parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=192,type=int,help="number of dims")
parser.add_argument('-e', '--encoder_layers', nargs='+', type=int, help="A list of dimensions for the encoder")
parser.add_argument('-d', '--decoder_layers', nargs='+', type=int, help="A list of dimensions for the decoder")
parser.add_argument('-model', '--model', choices=['6Net','12Net', '24Net', '48Net','gmm'], default='12Net')


args = parser.parse_args()
mgh= None

# nb_features=64
if args.encoder_layers:
    nb_features = '_'.join(map(str, args.encoder_layers))

ngpus =1#len(os.environ["CUDA_VISIBLE_DEVICES"])
print(f'using {ngpus} gpus')
if ngpus > 1:
    model_device = '/gpu:0'
    synth_device = '/gpu:1'
    synth_gpu = 1
    dev_str = ", ".join(map(str, range(ngpus)))
    print("dev_str:",dev_str)
else:
    model_device = '/gpu:0'
    synth_device = model_device
    synth_gpu = 0
    dev_str = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
print(f'model_device {model_device}, synth_device {synth_device}, dev_str {dev_str}')
print(f'physical GPU # is {os.getenv("SLURM_STEP_GPUS")}')

if args.num_dims==256:
    print("loading 256 fetus dataset.")
    feta = pathlib.Path('feta_resized_256')
    with open("params.json", "r") as json_file:
        config = json.load(json_file)
elif args.num_dims==192:
    print("loading 192 fetus dataset.")
    feta = pathlib.Path('feta_resized_192')
    with open("params_192.json", "r") as json_file:
        config = json.load(json_file)

feta_files = list(feta.glob('*.nii.gz'))


# feta_label_maps = [np.uint8(f.dataobj) for f in map(nib.load, feta_files)]


# if args.synth:
log_dir = 'logs/logs_cascade'
models_dir = 'models_cascade'


if args.model=='6Net':
    log_dir += '_6net'
    models_dir += '_6net'
elif args.model=='12Net':
    log_dir += '_12net'
    models_dir += '_12net'
elif args.model=='24Net':
    log_dir += '_24net'
    models_dir += '_24net'
elif args.model=='48Net':
    log_dir += '_48net'
    models_dir += '_48net'
elif args.model=='gmm':
    log_dir += '_gmm'
    models_dir += '_gmm' 


    
mgh_files = []
mgh_label_maps = []

# label_maps = feta_label_maps
# labels = np.unique(label_maps)
# num_labels=9
# in_shape = label_maps[0].shape
# print("in_shape",in_shape)



num_shapes = 8

subfolders = [f.name for f in os.scandir("validation") if f.is_dir()]

latest_weight = max(glob.glob(os.path.join(models_dir, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

if latest_weight:
    shutil.move(latest_weight, os.path.join(models_dir, 'weights_epoch_0.h5'))


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
else:
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

checkpoint_path=models_dir+'/weights_epoch_0.h5'

if not os.listdir(log_dir):
    initial_epoch = 0
    print("initial epoch is 0")
elif latest_weight:
    match = re.search(r'(\d+)', latest_weight)
    initial_epoch = int(match.group())
else:
    initial_epoch= args.initial_epoch




batch_size=args.batch_size
warp_max=2   
warp_max=2.5
warp_min=.5
warp_blur_min=np.array([2, 4, 8])
warp_blur_max=warp_blur_min*2
bias_blur_min=np.array([2, 4, 8])
bias_blur_max=bias_blur_min*2
initial_lr=args.learning_rate
nb_conv_per_level=args.nb_conv_per_level
# lr = args.learning_rate
nb_levels=5
conv_size=3
num_epochs=param_3d.epoch_num
# num_bg_labels=16
warp_fwhm_min=10
warp_fwhm_max=20
warp_min_shapes=10
warp_max_shapes=50
# in_shape=(dimx,dimy,dimz)
bg_brain = True

warp_max=2
warp_min=1
image_fwhm_min=20
image_fwhm_max=40
aff_shift=30
aff_rotate=180
aff_shear=0.1
blur_max=2.4
slice_prob=1
crop_prob=1
bias_min=0.01
bias_max=0.2
zero_background=args.zero_background
aff_scale=args.scale
up_scale=False


# Access the configuration
model1_config = config["brain"]
model2_config = config["body"]

model_shapes_config = config["shapes"]

# model3_config = config["labels_to_image_model"]


# if not args.sdt:
# model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}

model1 = create_model(model1_config)
model2 = create_model(model2_config)

# FeTA brain


#shapes



# Model

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=20, verbose=1, min_lr=1e-7)

weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=100)  # Save weights every 100 epochs

early_stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-4, verbose=1)



TB_callback = CustomTensorBoard(
    base_log_dir=log_dir,
    histogram_freq=1000,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
)



from tensorflow.keras.losses import binary_crossentropy


if __name__ == "__main__":
    en = args.encoder_layers
    de = args.decoder_layers
    random.seed(3000)
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    if args.model=='6Net':
        model_feta_config = config["feta_6"]

        brain_maps = [np.uint8(sf.load_volume(str(file_path)).resize(1.2).reshape([param_3d.img_size_6,]*3).data) for file_path in feta_files]
        brain_maps = [ndimage.gaussian_filter(img, sigma=(3, 3, 3), order=0) for img in brain_maps]

        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        
        model_shapes_config["in_shape"]=[ param_3d.img_size_6, param_3d.img_size_6, param_3d.img_size_6]
        model_feta_config["in_shape"]=[ param_3d.img_size_6, param_3d.img_size_6, param_3d.img_size_6]

        model3_config = config["labels_to_image_model_6"]
        model3_config["in_shape"]=[ param_3d.img_size_6, param_3d.img_size_6, param_3d.img_size_6]
        model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}

        model_shapes = create_model(model_shapes_config)
        model_feta = create_model(model_feta_config)
        
        labels_to_image_model = create_model(model3_config)

        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6, 1), nb_features=(en, de), batch_norm=False,
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        input_img = Input(shape=(param_3d.img_size_6,param_3d.img_size_6,param_3d.img_size_6,1))

        _, fg = model_feta(input_img)
        
        shapes = draw_shapessba(shape = (param_3d.img_size_6,)*3)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.uint8)
        
        _, bg = model_shapes(shapes[None,...,None])
        bg = shift_non_zero_elements(bg,8)    
        result = fg + bg * tf.cast(fg == 0,tf.int32)
        
        # result = random_selection_layer(bg,result,maxval=150)


        
        generated_img , y = labels_to_image_model(result)

        # generated_img = apply_gaussian_smoothing(generated_img)
        generated_img_norm = min_max_norm(generated_img)
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=param_3d.lr))
        
        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]

        gen = generator_brain(brain_maps)
    elif args.model=='12Net':
        model_feta_config = config["feta_12"]

        # brain_maps = [np.uint8(sf.load_volume(str(file_path)).reshape([param_3d.img_size_12,]*3).data) for file_path in feta_files]
        brain_maps = [np.uint8(sf.load_volume(str(file_path)).resize(1.5).reshape([param_3d.img_size_12,]*3).data) for file_path in feta_files]

        en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
        
        
        model_shapes_config["in_shape"]=[ param_3d.img_size_12, param_3d.img_size_12, param_3d.img_size_12]
        model_feta_config["in_shape"]=[ param_3d.img_size_12, param_3d.img_size_12, param_3d.img_size_12]

        model3_config = config["labels_to_image_model_12"]
        model3_config["in_shape"]=[ param_3d.img_size_12, param_3d.img_size_12, param_3d.img_size_12]
        model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}

        model_shapes = create_model(model_shapes_config)
        model_feta = create_model(model_feta_config)
        
        labels_to_image_model = create_model(model3_config)

        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12, 1), nb_features=(en, de), batch_norm=False,
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        input_img = Input(shape=(param_3d.img_size_12,param_3d.img_size_12,param_3d.img_size_12,1))

        _, fg = model_feta(input_img)
        
        shapes = draw_shapes(shape = (param_3d.img_size_12,)*3)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.uint8)
        
        _, bg = model_shapes(shapes[None,...,None])
        bg = shift_non_zero_elements(bg,8)    
        result = fg + bg * tf.cast(fg == 0,tf.int32)

        # result = random_selection_layer(bg,result,maxval=150)

        
        generated_img , y = labels_to_image_model(result)

        # generated_img = apply_gaussian_smoothing(generated_img)
        generated_img_norm = min_max_norm(generated_img)
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=param_3d.lr))
        
        # brain_maps = feta_label_maps
        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]

        gen = generator_brain(brain_maps)
        
    elif args.model=='24Net':
        brain_maps = [np.uint8(sf.load_volume(str(file_path)).resize(1.5).reshape([param_3d.img_size_24,]*3).data) for file_path in feta_files]

        # en = [32, 64, 64, 128, 128, 256, 512]  # Example encoder feature sizes
        # de = [512, 256, 128, 128, 64, 64, 32, 2]  # Example decoder feature sizes
        en = [32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,2]
        model_feta_config = config["feta_24"]
        model3_config = config["labels_to_image_model_24"]
        model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}
        
        model3_config["in_shape"]=[ param_3d.img_size_24, param_3d.img_size_24, param_3d.img_size_24]
        model_shapes_config["in_shape"]=[ param_3d.img_size_24, param_3d.img_size_24, param_3d.img_size_24]
        model_feta_config["in_shape"]=[ param_3d.img_size_24, param_3d.img_size_24, param_3d.img_size_24]
        model_shapes = create_model(model_shapes_config)
        model_feta = create_model(model_feta_config)
        
        labels_to_image_model = create_model(model3_config)

        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24, 1), nb_features=(en, de), batch_norm=False,
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        input_img = Input(shape=(param_3d.img_size_24,param_3d.img_size_24,param_3d.img_size_24,1))

        _, fg = model_feta(input_img)
        shapes = draw_shapes(shape = (param_3d.img_size_24,)*3,warp_min=12)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.uint8)
        
        _, bg = model_shapes(shapes[None,...,None])
        bg = shift_non_zero_elements(bg,8)    
        result = fg + bg * tf.cast(fg == 0,tf.int32)

        # result = random_selection_layer(bg,result,maxval=40)

        
        generated_img , y = labels_to_image_model(result)
        

        # generated_img = apply_gaussian_smoothing(generated_img)
        generated_img_norm = min_max_norm(generated_img)
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=param_3d.lr))
        
        # brain_maps = feta_label_maps
        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]

        gen = generator_brain(brain_maps)

    elif args.model=='48Net':
        brain_maps = [np.uint8(sf.load_volume(str(file_path)).resize(1.5).reshape([param_3d.img_size_48,]*3).data) for file_path in feta_files]
    
        # en = [32 ,32 ,64 ,64 ,64 ,64 ,64 ,128 ,128 ,256 ,256]
        # de = [256 ,256 ,128 ,128, 64 ,64 ,64, 64, 64, 32 ,32 ,2]
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]
        # en = [32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        # de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,2]

        model_feta_config = config["feta_48"]
        model3_config = config["labels_to_image_model_48"]
        model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}
        
        model3_config["in_shape"]=[ param_3d.img_size_48, param_3d.img_size_48, param_3d.img_size_48]
        model_shapes_config["in_shape"]=[ param_3d.img_size_48, param_3d.img_size_48, param_3d.img_size_48]
        model_feta_config["in_shape"]=[ param_3d.img_size_48, param_3d.img_size_48, param_3d.img_size_48]
        model_shapes = create_model(model_shapes_config)
        model_feta = create_model(model_feta_config)
        
        labels_to_image_model = create_model(model3_config)

        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48,1))

        _, fg = model_feta(input_img)
        shapes = draw_shapes(shape = (param_3d.img_size_48,)*3)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.uint8)
        tf.random.set_seed(42)

        _, bg = model_shapes(shapes[None,...,None])
        bg = shift_non_zero_elements(bg,8)    
        result = fg + bg * tf.cast(fg == 0,tf.int32)
        

    
        # result = random_selection_layer(bg,result)
        
        generated_img , y = labels_to_image_model(result)
        
        generated_img = apply_gaussian_smoothing(generated_img)
        generated_img_norm = min_max_norm(generated_img)
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=0.00001))

        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]
        gen = generator_brain(brain_maps)
    elif args.model=='gmm':
        feta = pathlib.Path('fetus_label_map')
        mgh_files = list(feta.glob('*.nii.gz'))
        brain_maps = [np.uint8(sf.load_volume(str(file_path)).data) for file_path in mgh_files]
    
        en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
        de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 32 ,32 ,16 ,16 ,2]

        model_feta_config = config["feta_48"]
        model3_config = config["labels_to_image_model_gmm"]
        model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}
        
        model3_config["in_shape"]=[ param_3d.img_size_48, param_3d.img_size_48, param_3d.img_size_48]
        
        labels_to_image_model = create_model(model3_config)

        unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48, 1), nb_features=(en, de), batch_norm=False,
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        input_img = Input(shape=(param_3d.img_size_48,param_3d.img_size_48,param_3d.img_size_48,1))
        
        generated_img , y = labels_to_image_model(input_img)
        generated_img = apply_gaussian_smoothing(generated_img)
        generated_img_norm = min_max_norm(generated_img)
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=param_3d.lr))

        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]
        gen = generator_brain_gmm(brain_maps,cube_size=param_3d.img_size_48)
    
    callbacks_list = [TB_callback, weights_saver,reduce_lr]
    
    if os.path.exists(checkpoint_path):
        print(checkpoint_path)
        combined_model.load_weights(checkpoint_path)
        print("Loaded weights from the checkpoint and continued training.")
    else:
        print("Checkpoint file not found.")
        
        # Train the model using the combined generator
    combined_model.fit(gen, epochs=num_epochs, batch_size=1, steps_per_epoch=100, validation_steps=5, callbacks=callbacks_list)

