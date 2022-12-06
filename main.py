## Load Packages
import numpy as np
import pandas as pd
import cv2
import os
import random

import sys
import config
sys.path.append('model_building/create_image_folders.py')
from model_building.create_image_folders import * 
sys.path.append('model_building/keras_model.py')
from model_building.cnn_model_keras import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import keras 
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from tensorflow.keras import layers
import warnings
warnings.simplefilter("ignore", UserWarning)

## Set paths
## Set paths
img_folder = os.path.join("data","ai_ready","images")
train_img = os.path.join("data","ai_ready","train_images")
val_img = os.path.join("data","ai_ready","val_images")
labels_image = os.path.join("data","ai_ready","x-ai_data.csv")
create_images =False

if __name__ == '__main__':
    tf.config.list_physical_devices()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices)>0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ## Move images to subfolders
    if create_images:
        subfolders(labels_image, img_folder, train_img, val_img)

    ## Model
    model = make_model(input_shape=config.image_size + (3,), num_classes=2)

    ## Train and Val dataset
    train_ds = train_set(train_img, config.image_size, config.batch_size)
    val_ds = val_set(val_img, config.image_size, config.batch_size)

    ## Train Model
    train_model(model, train_ds, val_ds, config.number_epochs)
