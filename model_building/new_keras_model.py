import config
import numpy as np
import pandas as pd

import keras
from keras.initializers import RandomNormal
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    LeakyReLU,
    BatchNormalization,
)

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
import config

import warnings

warnings.simplefilter("ignore", UserWarning)


def train_set(train_path, image_size, batch_size):
    """
    Create train dataset to train the model
    Input:
        train_path: path referring to where the train images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        train_generator: train set in Keras format
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range = [0.5, 2.0])

    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=image_size, batch_size=batch_size, class_mode="binary"
    )

    return train_generator


def val_set(val_path, image_size, batch_size):
    """
    Create val dataset to train the model
    Input:
        val_path: path referring to where the val images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        val_generator: train set in Keras format
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_generator = test_datagen.flow_from_directory(
        val_path, target_size=image_size, batch_size=batch_size, class_mode="binary"
    )

    return val_generator


def test_set(test_path, image_size, batch_size):
    """
    Create test dataset to test the model
    Input:
        test_path: path referring to where the test images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        test_generator: test set in Keras format
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    return test_generator


def keras_model(input_shape, train_g, val_g, batch_size, epochs, model_name):
    """
    function building a keras deep learning model for image recognition
    Inputs:
        - input_shape : tuple containing the dimensions of the pictures
        - train_g: training generator
        - val_g: validation generator
        - batch_size: batch size used for training
        - epochs: number of epochs for training
        - model_name: name to save the model
    Output:
        - keras_model : trained keras model
    """
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(256,256,3)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],)

    model.fit_generator(
        train_g,
        steps_per_epoch=1400 // batch_size,
        epochs=epochs,
        validation_data=val_g,
        validation_steps=400 // batch_size,
    )

    model.save(model_name + ".h5")