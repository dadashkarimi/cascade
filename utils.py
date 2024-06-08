import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite_sandbox.tf.utils.utils import morphology_3d

# from neurite_sandbox.tf.losses import dtrans_loss
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
from utils import *


import scipy.ndimage as ndi
from skimage.measure import regionprops, marching_cubes

def img2array(img,dim):
     
    if dim == param.img_size_12:    
        if img.size[0] != param.img_size_12 or img.size[1] != param.img_size_12:
            img = img.resize((param.img_size_12,param.img_size_12))
        img = np.asarray(img).astype(np.float32)/255 
    elif dim == param.img_size_24:
        if img.size[0] != param.img_size_24 or img.size[1] != param.img_size_24:
            img = img.resize((param.img_size_24,param.img_size_24))
        img = np.asarray(img).astype(np.float32)/255
    elif dim == param.img_size_48:
        if img.size[0] != param.img_size_48 or img.size[1] != param.img_size_48:
            img = img.resize((param.img_size_48,param.img_size_48))
        img = np.asarray(img).astype(np.float32)/255
    return img

def calib_box(result_box,result,img):
    

    for id_,cid in enumerate(np.argmax(result,axis=1).tolist()):
        s = cid / (len(param.cali_off_x) * len(param.cali_off_y))
        x = cid % (len(param.cali_off_x) * len(param.cali_off_y)) / len(param.cali_off_y)
        y = cid % (len(param.cali_off_x) * len(param.cali_off_y)) % len(param.cali_off_y) 
                
        s = param.cali_scale[s]
        x = param.cali_off_x[x]
        y = param.cali_off_y[y]
    
        
        new_ltx = result_box[id_][0] + x*(result_box[id_][2]-result_box[id_][0])
        new_lty = result_box[id_][1] + y*(result_box[id_][3]-result_box[id_][1])
        new_rbx = new_ltx + s*(result_box[id_][2]-result_box[id_][0])
        new_rby = new_lty + s*(result_box[id_][3]-result_box[id_][1])
        
        result_box[id_][0] = int(max(new_ltx,0))
        result_box[id_][1] = int(max(new_lty,0))
        result_box[id_][2] = int(min(new_rbx,img.size[0]-1))
        result_box[id_][3] = int(min(new_rby,img.size[1]-1))
        result_box[id_][5] = img.crop((result_box[id_][0],result_box[id_][1],result_box[id_][2],result_box[id_][3]))

    return result_box 

def NMS(box):
    
    if len(box) == 0:
        return []
    
    #xmin, ymin, xmax, ymax, score, cropped_img, scale
    box.sort(key=lambda x :x[4])
    box.reverse()

    pick = []
    x_min = np.array([box[i][0] for i in range(len(box))],np.float32)
    y_min = np.array([box[i][1] for i in range(len(box))],np.float32)
    x_max = np.array([box[i][2] for i in range(len(box))],np.float32)
    y_max = np.array([box[i][3] for i in range(len(box))],np.float32)

    area = (x_max-x_min)*(y_max-y_min)
    idxs = np.array(range(len(box)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap <= 1)))[0]+1)))
    
    return [box[i] for i in pick]

def sliding_window(img, thr, net, input_12_node):

    pyramid = tuple(pyramid_gaussian(img, downscale = param.downscale))
    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(param.pyramid_num):
        
        X = pyramid[scale]

        resized = Image.fromarray(np.uint8(X*255)).resize((int(np.shape(X)[1] * float(param.img_size_12)/float(param.face_minimum)), int(np.shape(X)[0]*float(param.img_size_12)/float(param.face_minimum))))
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]

        X = np.reshape(X,(1,img_row,img_col,param.input_channel))
        
        if img_row < param.img_size_12 or img_col < param.img_size_12:
            break
        
        #predict and get rid of boxes from padding
        win_num_row = math.floor((img_row-param.img_size_12)/param.window_stride)+1
        win_num_col = math.floor((img_col-param.img_size_12)/param.window_stride)+1

        result = net.prediction.eval(feed_dict={input_12_node:X})
        result_row = np.shape(result)[1]
        result_col = np.shape(result)[2]

        result = result[:,\
                int(math.floor((result_row-win_num_row)/2)):int(result_row-math.ceil((result_row-win_num_row)/2)),\
                int(math.floor((result_col-win_num_col)/2)):int(result_col-math.ceil((result_col-win_num_col)/2)),\
                :]

        feature_col = np.shape(result)[2]

        #feature_col: # of predicted window num in width dim
        #win_num_col: # of box(gt)
        assert(feature_col == win_num_col)

        result = np.reshape(result,(-1,1))
        result_id = np.where(result > thr)[0]
        
        #xmin, ymin, xmax, ymax, score
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)*param.window_stride
        detected_list_scale[:,1] = np.floor(result_id/feature_col)*param.window_stride
        detected_list_scale[:,2] = np.minimum(detected_list_scale[:,0] + param.img_size_12 - 1, img_col-1)
        detected_list_scale[:,3] = np.minimum(detected_list_scale[:,1] + param.img_size_12 - 1, img_row-1)

        detected_list_scale[:,0] = detected_list_scale[:,0] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,1] = detected_list_scale[:,1] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,2] = detected_list_scale[:,2] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,3] = detected_list_scale[:,3] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,4] = result[result_id,0]

        detected_list_scale = detected_list_scale.tolist()
       
        #xmin, ymin, xmax, ymax, score, cropped_img, scale
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale] for id_,elem in enumerate(detected_list_scale)]
        
        if len(detected_list_scale) > 0:
            detected_list[scale] = detected_list_scale 
            
    detected_list = [elem for elem in detected_list if type(elem) != int]
    result_box = [detected_list[i][j] for i in xrange(len(detected_list)) for j in xrange(len(detected_list[i]))]
    
    return result_box

def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize
    
def load_validation_data(validation_folder_path,dim_):
    subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
    image_mask_pairs = []

    for folder in subfolders:
        folder_path = os.path.join(validation_folder_path, folder)
        filename = os.path.join(folder_path,"image.nii.gz")
        mask_filename = os.path.join(folder_path,"manual.nii.gz")
        image = sf.load_volume(filename)
        new_voxsize = [dynamic_resize(image)]*3

        orig_voxsize = image.geom.voxsize
        crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        crop_img = crop_img.resize(new_voxsize, method="linear").reshape([dim_, dim_, dim_])
        crop_data = crop_img.data
        mask = sf.load_volume(mask_filename).resize([orig_voxsize[0],orig_voxsize[1],1
                                                    ], method="linear")
        mask = mask.resize(new_voxsize).reshape([dim_, dim_, dim_, 1])
        mask.data[mask.data != 0] = 1
        mask.data = tf.cast(mask.data, tf.int32)
        image_mask_pairs.append((crop_data,mask.data))
    return image_mask_pairs


                
def generator_brain_window_Net(label_maps,img_size):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    
    while True:
        fg = rand.choice(label_maps)
        yield fg[None,...,None]
        
import tensorflow as tf
from tensorflow.keras.layers import Layer

        

def generate_position_map(volume_shape, patch_size, stride):
    positions = []
    indices = []
    index = 0
    for x1 in range(0, volume_shape[0] - patch_size + stride, stride):
        for y1 in range(0, volume_shape[1] - patch_size + stride, stride):
            for z1 in range(0, volume_shape[2] - patch_size + stride, stride):
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                z2 = z1 + patch_size
                positions.append((x1, y1, z1, x2, y2, z2))
                indices.append(index)
                index += 1
    return positions, indices
import tensorflow as tf

def generate_position_map_tf(volume_shape, patch_size, stride):
    positions = []
    indices = []

    x_range = tf.range(0, volume_shape[0] - patch_size + 4, stride, dtype=tf.int32)
    y_range = tf.range(0, volume_shape[1] - patch_size + 4, stride, dtype=tf.int32)
    z_range = tf.range(0, volume_shape[2] - patch_size + 4, stride, dtype=tf.int32)

    index = 0
    for x1 in x_range:
        for y1 in y_range:
            for z1 in z_range:
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                z2 = z1 + patch_size
                positions.append((x1, y1, z1, x2, y2, z2))
                indices.append(index)
                index += 1

    # positions = tf.constant(positions, dtype=tf.int32)
    # indices = tf.constant(indices, dtype=tf.int32)

    return positions, indices


def convert_to_binary_vector(pattern_list):
    binary_vector = [1 if pattern is not None else 0 for pattern in pattern_list]
    return binary_vector

# def find_bounding_box(mask, cube_size=32):
#     non_zero_coords = np.argwhere(mask)
#     min_coords = non_zero_coords.min(axis=0)
#     max_coords = non_zero_coords.max(axis=0)
#     center = (min_coords + max_coords) // 2
#     half_size = cube_size // 2
#     x1 = max(center[0] - half_size, 0)
#     y1 = max(center[1] - half_size, 0)
#     z1 = max(center[2] - half_size, 0)
#     x2 = min(center[0] + half_size, mask.shape[0])
#     y2 = min(center[1] + half_size, mask.shape[1])
#     z2 = min(center[2] + half_size, mask.shape[2])
#     return x1, y1, z1, x2, y2, z2

def find_bounding_box(mask, cube_size=32, max_dim=192):
    non_zero_coords = np.argwhere(mask)
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    center = (min_coords + max_coords) // 2
    half_size = cube_size // 2

    # Ensure that the bounding box does not exceed the maximum dimension
    x1 = max(center[0] - half_size, 0)
    y1 = max(center[1] - half_size, 0)
    z1 = max(center[2] - half_size, 0)

    x2 = min(center[0] + half_size, min(mask.shape[0], max_dim))
    y2 = min(center[1] + half_size, min(mask.shape[1], max_dim))
    z2 = min(center[2] + half_size, min(mask.shape[2], max_dim))

    return x1, y1, z1, x2, y2, z2

def find_random_bounding_box(input_volume, cube_size=32, margin=8, full_random=False):
    if full_random:
        return find_full_random_bounding_box(input_volume,cube_size)
    non_zero_coords = np.argwhere(input_volume)
    min_coords = non_zero_coords.min(axis=0) - margin
    max_coords = non_zero_coords.max(axis=0) + margin
    random_coords = [np.random.randint(min_c, max_c - cube_size + 1) for min_c, max_c in zip(min_coords, max_coords)]
    x1, y1, z1 = random_coords
    x2, y2, z2 = x1 + cube_size, y1 + cube_size, z1 + cube_size
    return x1, y1, z1, x2, y2, z2
    
def find_full_random_bounding_box(input_volume, cube_size=32):
    x_max, y_max, z_max = input_volume.shape
    x1 = np.random.randint(0, x_max - cube_size)
    y1 = np.random.randint(0, y_max - cube_size)
    z1 = np.random.randint(0, z_max - cube_size)
    x2 = x1 + cube_size
    y2 = y1 + cube_size
    z2 = z1 + cube_size
    return x1, y1, z1, x2, y2, z2
    
def extract_cube(input_volume, x1, y1, z1, x2, y2, z2, cube_size=32):
    cube = np.zeros((cube_size, cube_size, cube_size))
    x_size, y_size, z_size = x2 - x1, y2 - y1, z2 - z1
    cube[:x_size, :y_size, :z_size] = input_volume[x1:x2, y1:y2, z1:z2]
    return cube

# def extract_cube(input_volume, x1, y1, z1, x2, y2, z2,cube_size=32):
#     cube_size = (x2 - x1, y2 - y1, z2 - z1)
#     cube = np.zeros(cube_size)
#     x_start, x_end = max(0, -x1), min(cube_size[0], input_volume.shape[0] - x1)
#     y_start, y_end = max(0, -y1), min(cube_size[1], input_volume.shape[1] - y1)
#     z_start, z_end = max(0, -z1), min(cube_size[2], input_volume.shape[2] - z1)
#     cube[x_start:x_end, y_start:y_end, z_start:z_end] = input_volume[x1+x_start:x1+x_end, y1+y_start:y1+y_end, z1+z_start:z1+z_end]
#     return cube


# def extract_cube(input_volume, x1, y1, z1, x2, y2, z2, cube_size=32):
#     max_cube_size = min(cube_size, min(x2 - x1, y2 - y1, z2 - z1, 192))
#     cube = np.zeros((max_cube_size, max_cube_size, max_cube_size))
#     x_size, y_size, z_size = min(x2 - x1, max_cube_size), min(y2 - y1, max_cube_size), min(z2 - z1, max_cube_size)
#     cube[:x_size, :y_size, :z_size] = input_volume[x1:x1 + x_size, y1:y1 + y_size, z1:z1 + z_size]
#     return cube
    
def extract_fragments(input_volume, positions):
    fragments = []
    for pos in positions:
        x1, y1, z1, x2, y2, z2 = pos
        fragment = input_volume[x1:x2, y1:y2, z1:z2]
        fragments.append(fragment)
    return np.stack(fragments)

def extract_single_fragment(input_volume, position):
    x1, y1, z1, x2, y2, z2 = position
    fragment = input_volume[x1:x2, y1:y2, z1:z2]
    return fragment

def random_select_tensor(bg, result,maxval=2):
    random_val = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
    return tf.cond(tf.equal(random_val, 0), lambda: bg, lambda: result)


def random_selection_layer(bg, result,maxval=2):
    return Lambda(lambda x: random_select_tensor(x[0], x[1],maxval))([bg, result])
    

def apply_gaussian_smoothing(tensor,sigma = 1.0,kernel_size = 3):
    kernel = tf.exp(-0.5 * tf.square(tf.linspace(-1.0, 1.0, kernel_size)) / sigma**2)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1, 1])
    kernel = tf.tile(kernel, [1, kernel_size, 1, 1, 1])
    kernel = tf.tile(kernel, [1, 1, kernel_size, 1, 1])
    return tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
def shift_non_zero_elements(bg, shift_value):
    non_zero_mask = tf.not_equal(bg, 0)
    shifted_non_zero_elements = tf.where(non_zero_mask, bg + shift_value, bg)
    return shifted_non_zero_elements
    
def generator_brain_label_maps(brain_maps):
    label_maps = np.asarray(brain_maps)
    for fg in label_maps:
        yield fg[None, ..., None]

def create_model(model_config):
    model_config_ = model_config.copy()
    # return model_3d.labels_to_image_new(**model_config_)
    return ne.models.labels_to_image_new(**model_config_)

def soft_dice(a, b):
    dim = len(a.shape) - 2
    space = list(range(1, dim + 1))

    top = 2 * tf.reduce_sum(a * b, axis=space)
    bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)
    
    out = tf.divide(top, bot + 1e-6)
    return -tf.reduce_mean(out)
    
def create_window_model(positions, window_size, model_config):
    model_config_ = model_config.copy()
    model_config["window_size"]=window_size
    model_config["positions"]=positions
    return ne.models.labels_to_windows_new(**model_config_)

import tensorflow as tf
from tensorflow.keras.layers import Layer


import numpy as np

class GenerateLabelsLayer(tf.keras.layers.Layer):
    def __init__(self, volume_shape, positions, patch_size, stride, **kwargs):
        super(GenerateLabelsLayer, self).__init__(**kwargs)
        self.volume_shape = volume_shape
        self.positions = positions

    def call(self, y):
        labels = []
        # print(y.shape,y)
        brain_indices = np.where(y > 0)
        print(brain_indices.shape)
        for pos in self.positions:
            x1, y1, z1, x2, y2, z2 = pos
            condition = (
                x1 >= brain_indices[0].min() and
                y1 >= brain_indices[1].min() and
                z1 >= brain_indices[2].min() and
                x2 <= brain_indices[0].max() and
                y2 <= brain_indices[1].max() and
                z2 <= brain_indices[2].max()
            )
            labels.append(1 if condition else 0)
        return np.array(labels)




@tf.function
def process_fragment(x1, y1, z1, x2, y2, z2, img_size, y):
    if tf.executing_eagerly():
        fragment = tf.zeros((img_size, img_size, img_size), dtype=tf.float32)
        label = tf.constant(0, dtype=tf.float32)
    else:
        mask = y
        brain_indices = tf.where(mask > 0)
        condition = (
            x1 >= tf.reduce_min(brain_indices[:, 0]) and
            y1 >= tf.reduce_min(brain_indices[:, 1]) and
            z1 >= tf.reduce_min(brain_indices[:, 2]) and
            x2 <= tf.reduce_max(brain_indices[:, 0]) and
            y2 <= tf.reduce_max(brain_indices[:, 1]) and
            z2 <= tf.reduce_max(brain_indices[:, 2])
        )
        label = tf.cond(condition, lambda: tf.constant(1), lambda: tf.constant(0))
    
    return fragment, label


def produce_labels(y_shape, positions):
    labels = tf.zeros(y_shape[1], dtype=tf.int32)
    return labels


def generator_fragments_and_labels(label_map, model_feta, model_shapes, labels_to_image_model, img_size, batch_size):
    input_img = Input(shape=(192, 192, 192, 1))
    _, fg = model_feta(input_img)
    shapes = draw_shapes_easy(shape=(192,) * 3)
    shapes = tf.squeeze(shapes)
    shapes = tf.cast(shapes, tf.uint8)
    _, bg = model_shapes(shapes[None, ..., None])
    bg = bg + 8
    fg_inner = fg[0, ..., 0]
    bg_inner = tf.reshape(bg[0, ..., 0], fg_inner.shape)
    mask = tf.cast(tf.equal(fg_inner, 0), fg_inner.dtype)
    result = fg_inner + bg_inner * mask

    generated_img, y = labels_to_image_model(result[None, ..., None])
    
    if len(generated_img) == 0:
        yield tf.zeros((1, img_size, img_size, img_size, 1), dtype=tf.float32), tf.constant([0], dtype=tf.float32)
    else:
        for img, label in zip(generated_img, y):
            yield img[None, ...], tf.constant([label], dtype=tf.float32)
            
    # for i in range(0, len(positions), batch_size):
    #     fragment_batch = []
    #     label_batch = []
        
    #     for pos in positions[i:i+batch_size]:
    #         x1, y1, z1, x2, y2, z2 = pos
    #         # fragment = generated_img[0, x1:x2, y1:y2, z1:z2, 0]
    #         fragment, label = process_fragment(x1, y1, z1, x2, y2, z2, img_size, y)
    #         # if tf.executing_eagerly():
    #         #     fragment = tf.zeros((img_size, img_size, img_size), dtype=tf.float32)
    #         #     label = tf.constant(0,dtype=tf.float32)
    #         # else:
    #         #     mask = y
    #         #     brain_indices = tf.where(mask > 0)
    #         #     print("#####",tf.reduce_sum(brain_indices))
    #         #     condition = (
    #         #         x1 >= tf.reduce_min(brain_indices[:, 0]) and
    #         #         y1 >= tf.reduce_min(brain_indices[:, 1]) and
    #         #         z1 >= tf.reduce_min(brain_indices[:, 2]) and
    #         #         x2 <= tf.reduce_max(brain_indices[:, 0]) and
    #         #         y2 <= tf.reduce_max(brain_indices[:, 1]) and
    #         #         z2 <= tf.reduce_max(brain_indices[:, 2])
    #         #     )
    #         #     label = tf.cond(condition, lambda: tf.constant(1), lambda: tf.constant(0))
    #         fragment_batch.append(fragment)
    #         label_batch.append(label)
    #     yield K.constant(tf.stack(fragment_batch)), K.constant(tf.stack(label_batch))
    # for i, pos in enumerate(positions):
    #     x1, y1, z1, x2, y2, z2 = pos
    #     fragment = generated_img[0, x1:x2, y1:y2, z1:z2, :]
    #     yield fragment[None,...], labels[i][None,...]

# Create a combined generator
def combined_generator_12Net(brain_maps, model_feta,model_shapes,labels_to_image_model,batch_size):
    for label_map in generator_brain_label_maps(brain_maps):
        for fragment, label in generator_fragments_and_labels(label_map, model_feta,model_shapes,labels_to_image_model,param_3d.img_size_12 ,batch_size):
            yield fragment, label
            

def gen_fragments(input_volume, patch_size,y):
    fragments = []
    # labels = generate_labels((192,192,192), positions, indices, param_3d.img_size_24, 4, input_volume)
    positions, indices = generate_position_map((192,192,192), param_3d.img_size_12, 4)

    for i in range(len(positions)):
        x1, y1, z1, x2, y2, z2 = positions[i]
        fragment = input_volume[0, x1:x2, y1:y2, z1:z2, :]
        yield fragment
    
def calculate_iou(window, mask):
    xmin, ymin, zmin, xmax, ymax, zmax = window
    window_mask = np.zeros_like(mask)
    window_mask[xmin:xmax,  ymin:ymax, zmin:zmax] = 1

    # mask_non_zero = mask > 0
    intersection = np.logical_and(mask, window_mask).sum()
    box_volume = np.sum(window_mask)
    iou = intersection / box_volume if box_volume != 0 else 0
    return iou

def draw_shapes_easy(
    shape,
    label_min=8,
    label_max=16,
    fwhm_min=32,
    fwhm_max=128,
    dtype=None,
    seed=None,
    **kwargs,
):
    # Data types.
    type_f = tf.float32
    type_i = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)

    # Images and transforms.
    out = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 2),
        fwhm_min=fwhm_min,
        fwhm_max=fwhm_max,
        isotropic=False,
        batched=False,
        featured=True,
        seed=seed,
        dtype=type_f,
        reduce=tf.math.reduce_max,
    )
    out = ne.utils.minmax_norm(out)

    num_label = tf.random.uniform(shape=(), minval=label_min, maxval=label_max + 1, dtype=type_i)
    out *= tf.cast(num_label, type_f)
    out = tf.cast(out, type_i)

    # Random relabeling. For less rare marginal label values.
    def reassign(x, max_in, max_out):
        lut = tf.random.uniform(shape=[max_in + 1], maxval=max_out, dtype=type_i)
        return tf.gather(lut, indices=x)

    # Add labels to break concentricity.
    a = reassign(out[..., 0:1], max_in=num_label, max_out=num_label)
    b = reassign(out[..., 1:2], max_in=num_label, max_out=num_label)
    out = reassign(a + b, max_in=2 * num_label - 2, max_out=num_label)
    # out = out[None,...]
    return tf.cast(out, dtype) if out.dtype != dtype else out

def draw_shapes(
    shape,
    num_label=16,
    warp_min=1,
    warp_max=20,
    dtype=None,
    seed=None,
    image_fwhm_min=20,
    image_fwhm_max=40,
    warp_fwhm_min=40,
    warp_fwhm_max=80,
):
    # Data types.
    type_fp = tf.float16
    type_int = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)
    
    # Randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(isotropic=False, batched=False, featured=True, seed=seed(), dtype=type_fp, reduce=tf.math.reduce_max)
    
    # Images and transforms.
    v = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 1),
        fwhm_min=image_fwhm_min, fwhm_max=image_fwhm_max, **prop(),
    )
    
    t = ne.utils.augment.draw_perlin_full(
        shape=(*shape, len(shape)), noise_min=warp_min, noise_max=warp_max,
        fwhm_min=warp_fwhm_min, fwhm_max=warp_fwhm_max, **prop(),
    )
    
    # Application and background.
    v = ne.utils.minmax_norm(v)
    v = vxm.utils.transform(v, t, fill_value=0)
    v = tf.math.ceil(v * (num_label - 1))

    return tf.cast(v, dtype) if v.dtype != dtype else v
    
def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize


class PeriodicWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=200, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        # Save the weights every `save_freq` epochs
        if (epoch + 1) % self.save_freq == 0:
            weights_path = os.path.join(self.filepath, f"weights_epoch_{epoch + 1}.h5")
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")
    
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, base_log_dir, **kwargs):
         super(CustomTensorBoard, self).__init__(**kwargs)
         self.base_log_dir = base_log_dir

    def on_epoch_begin(self, epoch, logs=None):
        # if epoch % self.histogram_freq == 0:  # Check if it's the start of a new set of 50 epochs
        self.log_dir = self.base_log_dir
        super().set_model(self.model)

def generator_brain(label_maps):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        yield fg[None, ..., None]

def get_brain(a):
    a_copy = np.copy(a)
    for i in range(len(a)):
        a_copy[i][a[i] >7 ] = 0
    return a_copy
    
def generator_brain_gmm(label_maps,cube_size=128):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        fg = extract_centered_cube(fg,cube_size=cube_size)
        yield fg[None, ..., None]


def extract_random_cube(input_volume, cube_size=32):    
    x_max, y_max, z_max = input_volume.shape
    x1 = np.random.randint(0, x_max - cube_size + 1)
    y1 = np.random.randint(0, y_max - cube_size + 1)
    z1 = np.random.randint(0, z_max - cube_size + 1)
    
    x2 = x1 + cube_size
    y2 = y1 + cube_size
    z2 = z1 + cube_size
    cube = input_volume[x1:x2, y1:y2, z1:z2]
    return cube

def find_center_of_labels(input_volume, labels):
    coords = np.argwhere(np.isin(input_volume, labels))
    return np.mean(coords, axis=0).astype(int)

def extract_centered_cube(input_volume, cube_size=32):
    labels = [1, 2, 3, 4, 5, 6, 7]
    center_coords = find_center_of_labels(input_volume, labels)
    
    x_max, y_max, z_max = input_volume.shape
    half_size = cube_size // 2
    
    x1 = max(0, center_coords[0] - half_size)
    y1 = max(0, center_coords[1] - half_size)
    z1 = max(0, center_coords[2] - half_size)
    
    x2 = min(x_max, x1 + cube_size)
    y2 = min(y_max, y1 + cube_size)
    z2 = min(z_max, z1 + cube_size)
    
    cube = np.zeros((cube_size, cube_size, cube_size))
    cube[:x2-x1, :y2-y1, :z2-z1] = input_volume[x1:x2, y1:y2, z1:z2]
    
    return cube
    
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=1e-4, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if current < self.value:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping triggered. {self.monitor} reached {current}")
            self.model.stop_training = True

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize

def load_val(validation_folder_path):

    latest_images=[]
    latest_masks=[]
    b2_images=[]
    b3_images=[]
    b2_masks=[]
    b3_masks=[]

    subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
    
    for folder in subfolders:
        mom_str = folder.split("_")[1]
        if mom_str.isdigit():
            mom = int(mom_str)
        else:
            mom=0
        
        folder_path = os.path.join(validation_folder_path, folder)
        filename = os.path.join(folder_path,"image.nii.gz")
        mask_filename = os.path.join(folder_path,"manual.nii.gz")
        image = sf.load_volume(filename)
        
        new_voxsize = [dynamic_resize(image)]*3
            
        orig_voxsize = image.geom.voxsize
        crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        crop_img = crop_img.resize(new_voxsize, method="linear").reshape([192, 192, 192])
        crop_data = crop_img.data
        
        mask = sf.load_volume(mask_filename).resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        mask = mask.resize(new_voxsize).reshape([192, 192, 192, 1])
        mask.data[mask.data != 0] = 1

        
        # if abs(w-initial_epoch)<step_size:
        if mom<100:
            b2_images.append(crop_img)
            b2_masks.append(mask)
        else:
            b3_images.append(crop_img)
            b3_masks.append(mask)
        latest_images.append(crop_img)
        latest_masks.append(mask)
            

    return latest_images, latest_masks , b2_images, b3_images, b2_masks, b3_masks

def find_largest_component(mask):
    labeled_mask, num_features = ndi.label(mask)
    largest_component = None
    max_area = 0

    for region in regionprops(labeled_mask):
        if region.area > max_area:
            max_area = region.area
            largest_component = (labeled_mask == region.label)

    return largest_component if largest_component is not None else np.zeros_like(mask)