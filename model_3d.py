import tensorflow as tf
import param_3d
import numpy as np

import tensorflow as tf

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from tensorflow.keras import layers, models, optimizers

def unet_model(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = inputs
    
    # Encoder
    for filters in [16, 32, 64]:
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Bottleneck
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Add batch normalization
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)  # Add batch normalization
    
    # Decoder
    for filters in [64, 32, 16]:
        x = layers.Conv3DTranspose(filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)  # Add batch normalization
    
    outputs = layers.Conv3D(2, (1, 1, 1), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model



class detect_12Net(tf.keras.Model):
    def __init__(self):
        super(detect_12Net, self).__init__()
        self.conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# class detect_12Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_12Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x

# class detect_24Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_24Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=3, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x

class detect_24Net(tf.keras.Model):
    def __init__(self):
        super(detect_24Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv4 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
        self.conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='valid')
        self.conv10 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.flatten(x)
        return x


# class detect_48Net(tf.keras.Model):
#     def __init__(self):
#         super(detect_48Net, self).__init__()
#         self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
#         self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=3, padding='same')
#         self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
#         self.conv3 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
#         self.flatten = tf.keras.layers.Flatten()

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         return x


class detect_48Net(tf.keras.Model):
    def __init__(self):
        super(detect_48Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv4 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='valid')
        self.conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='valid')
        self.conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='valid')
        self.conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='valid')
        self.conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='valid')
        self.conv10 = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.flatten(x)
        return x


# Define the model
class calib_12Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_12Net, self).__init__()
        self.conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class calib_24Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_24Net, self).__init__()
        self.conv1 = Conv3D(32, (5, 5, 5), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class calib_48Net(tf.keras.Model):
    def __init__(self, num_classes=135):
        super(calib_48Net, self).__init__()
        self.conv1 = Conv3D(64, (5, 5, 5), activation='relu', padding='same')
        self.pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')
        self.conv2 = Conv3D(64, (5, 5, 5), activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Define weight and bias initialization functions
def weight_variable(shape, name):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv3d(x, W, stride, pad="SAME"):
    return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=pad)

def max_pool3d(x, ksize, stride):
    return tf.nn.max_pool3d(x, ksize=[1, ksize, ksize, ksize, 1], strides=[1, stride, stride, stride, 1], padding="SAME")

