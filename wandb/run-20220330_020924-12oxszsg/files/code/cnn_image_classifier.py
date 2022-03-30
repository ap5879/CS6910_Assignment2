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
from wandb.keras import WandbCallback


# def read_config(json_path):
#     with open(json_path, "r") as fp:
#         json_dict = json.load(fp)
#         return json_dict

def build_cnn_model(num_cnn_layers, num_filters, kernel_size, activation_cnn, activation_fc, dense_neurons, img_size, pool_kernel, dropout):

    #Model Initialization
    model = Sequential()

    #Add subsequent convolution layers
    for i in range(num_cnn_layers):
        print("Adding layer {} to the network".format(i))
        model.add(Conv2D(num_filters[i], (kernel_size, kernel_size), padding= "valid"))
        model.add(Activation(activation_cnn))
        model.add(MaxPooling2D(pool_size=(pool_kernel, pool_kernel)))
        model.add(Dropout(dropout))

    #Fully connected layers
    model.add(Flatten())
    model.add(Dense(dense_neurons, activation = activation_fc))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation= "softmax"))

    input_shape = (None, img_size[0], img_size[1], 3)

    model.build(input_shape)

    model.summary()

    return model

def data_generation(data_aug_flag, data_split, batch_size):
    #Data Generation
    if data_aug_flag == "True":
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
    return train_generator, validation_generator, test_generator

def net_train(json_dict, img_size):

    # try:
    #     json_dict = read_config(json_path)
    # except:
    #     if len(sys.argv) < 2:
    #         print("User Error: No config file provided...using default config....")
    #         json_dict = read_config(".\\cnn_config.json")
    
    #Read Configs
    learning_rate = json_dict["learning_rate"]
    epochs = json_dict["epochs"]
    batch_size = json_dict["batch_size"]
    data_split = json_dict["data_split"]
    data_augmentation = json_dict["data_augmentation"]
    dropout = json_dict["dropout"]

    num_cnn_layers = json_dict["num_cnn_layers"]
    num_filters = json_dict["num_filters"]
    kernel_size = json_dict["kernel_size"]
    pool_kernel = json_dict["pool_kernel"]
    optimiser = json_dict["optimiser"]
    activation_cnn = json_dict["activation_cnn"]
    activation_fc = json_dict["activation_fc"]
    dense_neurons = json_dict["dense_neurons"]

    model = build_cnn_model(num_cnn_layers, num_filters, kernel_size, activation_cnn, activation_fc, dense_neurons, img_size, pool_kernel, dropout)

    config_default = dict(json_dict)
    wandb.init(project = 'CS6910-Assignment2-CNNs', config = config_default, entity='anuj-sougat')
    
    CONFIG = wandb.config
    wandb.run.name = "Assignment2_NumFltrs_" + str(CONFIG.num_filters) + "_DN_" + str(CONFIG.dense_neurons) + "_OPT_" + CONFIG.optimiser + "_DO_" + str(CONFIG.dropout) + "_BS_" + str(CONFIG.batch_size)
    
    train_generator, validation_generator, test_generator = data_generation(data_augmentation, data_split, batch_size)

    model.compile(
        optimizer= optimiser,
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits= True),
        metrics = ["accuracy"]
    )

    history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples//batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples//batch_size,
        epochs = epochs,
        callbacks = [WandbCallback]
    )
    model.save(".\\TrainedModel\\" + wandb.run.name)
    wandb.finish()

    return model, history

if __name__ == "__main__":

    #json_path = sys.argv[1]

    json_dict = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 32,
    "data_split": 0.8,
    "data_augmentation": "False",
    "dropout": 0.2,
    "num_cnn_layers": 5, 
    "kernel_size": 3,
    "num_filters": [16, 32, 64, 128, 256],
    "pool_kernel": 2,
    "optimiser": "adam",
    "weight_initialisation": "random",
    "activation_cnn": "relu",
    "activation_fc": "sigmoid",
    "dense_neurons": 100
    }

    img_size = (128, 128)

    sweep_config = {
    "name": "Bayesian Sweep",
    "method": "bayes",
    "metric":{
        "name": "val_accuracy",
        "goal": "maximize"
        },
    'early_terminate': {
        'type':'hyperband',
        'min_iter': [3],
        's': [2]
        },
    "parameters":{
        "activation_cnn": {
            "values": ["relu", "elu", "selu"]
            },
        "batch_size":{
            "values": [32, 64, 128]
            },
        "data_augmentation": {
            "values": ["True", "False"]
            },
        "epochs":{
            "values": [20, 30, 40, 50, 60]
            },
        "learning_rate": {
            "values": [0.0001, 0.005, 0.001]
            },
        "num_filters": {
            "values": [
                [16, 32, 64, 128, 256],
                [32, 64, 128, 256, 512],
                [32, 32, 32, 32, 32],
                [512, 256, 128, 64, 32],
                ]
            }
        }
    }

    start_time = time.time()

    net_train(json_dict, img_size)
    sweep_id = wandb.sweep(sweep_config, project='CS6910-Assignment2-CNNs', entity='anuj-sougat')

    wandb.agent(sweep_id, net_train, count = 100)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Total time taken:", elapsed_time)