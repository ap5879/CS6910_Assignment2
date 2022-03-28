import wandb
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
import pandas as pd
import numpy as np
import json
import sys
import time
import os

def read_config(json_path):
    with open(json_path, "r") as fp:
        json_dict = json.load(fp)
        return json_dict

def cnn_model(num_cnn_layers, num_filters, kernel_size, activation_cnn, activation_fc, dense_neurons, img_size, pool_kernel):

    #Model Initialization
    model = Sequential()

    #Add subsequent convolution layers
    for i in range(num_cnn_layers):
        print("Adding {}th layer to the network".format(i))
        model.add(Conv2D(num_filters[i], (kernel_size, kernel_size), padding= "valid"))
        model.add(Activation(activation_cnn))
        model.add(MaxPooling2D(pool_size=(pool_kernel, pool_kernel)))

    #Fully connected layers
    model.add(Flatten())
    model.add(Dense(dense_neurons, activation = activation_fc))
    model.add(Dense(10, activation= "softmax"))

    #input_shape = (None, img_size[0], img_size[1], 3)
    input_shape = (None, 128, 128, 3)

    model.build(input_shape)

    model.summary()

    return model

def net_train():
    pass

if __name__ == "__main__":
    try:
        json_path = sys.argv[1]
        json_dict = read_config(json_path)
    except:
        if len(sys.argv) < 2:
            print("User Error: No config file provided...using default config....")
            json_dict = read_config(".\\cnn_config.json")

    
    #print(json_dict)

    #Read Configs
    learning_rate = json_dict["training_params"]["learning_rate"]
    epochs = json_dict["training_params"]["epochs"]
    batch_size = json_dict["training_params"]["batch_size"]
    data_split = json_dict["training_params"]["data_split"]
    data_augmentation = json_dict["training_params"]["data_augmentation"]

    num_cnn_layers = json_dict["architecture_params"]["num_cnn_layers"]
    num_filters = json_dict["architecture_params"]["num_filters"]
    kernel_size = json_dict["architecture_params"]["kernel_size"]
    pool_kernel = json_dict["architecture_params"]["pool_kernel"]
    optimiser = json_dict["architecture_params"]["optimiser"]
    activation_cnn = json_dict["architecture_params"]["activation_cnn"]
    activation_fc = json_dict["architecture_params"]["activation_fc"]
    dense_neurons = json_dict["architecture_params"]["dense_neurons"]

    img_size = (128, 128)

    #Data Generation
    if data_augmentation == "True":
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1.0/255,
            validation_split = data_split,
            shear_range = 0.2,
            zoom_range = 0.2,
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 15,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False
        )
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1.0/255,
            validation_split = data_split
        )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)

    train_generator = train_datagen.flow_from_directory(
        ".\\inaturalist_12k\\train",
        target_size = img_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 123
    )

    validation_generator = train_datagen.flow_from_directory(
        ".\\inaturalist_12k\\train",
        target_size = img_size,
        subset = "validation",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 123
    )

    test_generator = test_datagen.flow_from_directory(
        ".\\inaturalist_12k\\test",
        target_size = img_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 123
    )

    model = cnn_model(num_cnn_layers, num_filters, kernel_size, activation_cnn, activation_fc, dense_neurons, img_size, pool_kernel)

    start_time = time.time()

    model.compile(
        optimizer= "adam",
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits= True),
        metrics = ["accuracy"]
    )

    history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples//batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples//batch_size,
        epochs = epochs
    )

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Total time taken:", elapsed_time)