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
from model_building.keras_model import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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
img_folder = "/home/jovyan/my_work/QB/image/images/"
train_img = "/home/jovyan/my_work/QB/image/train/"
val_img = "/home/jovyan/my_work/QB/image/val/"
labels_image = "data/ai_ready/x-ai_data.csv"

if __name__ == '__main__':
    tf.config.list_physical_devices()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ## Move images to subfolders
    subfolders(labels_image, img_folder, train_img, val_img)

    ## Model
    model = make_model(input_shape=config.image_size + (3,), num_classes=2)

    ## Train and Val dataset
    train_ds = train_set(train_img, config.image_size, config.batch_size)
    val_ds = val_set(val_img, config.image_size, config.batch_size)

    ## Train Model
    train_model(model, model, train_ds, val_ds, config.number_epochs)
