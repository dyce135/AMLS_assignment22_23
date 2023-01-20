# Run this file for model training
# Import packages
import pandas as pd
import os
from os.path import join, exists

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import celebGender as cg
import numpy as np

# Read labels
train_labels = pd.read_csv(join(script_dir, "Datasets/celeba/labels.csv"), sep='\t')
train_genders = train_labels["gender"]
train_genders = train_genders.replace(-1, 0)
print(train_genders.head())
train_dir = join(script_dir, "Datasets/celeba/img")

# Resize and reorganise images to classes
if not exists(join(script_dir, "Datasets/celeba_resized")):
    print("Resizing training data...")
    os.mkdir(join(script_dir, "Datasets/celeba_resized"))
    os.mkdir(join(script_dir, "Datasets/celeba_resized/male"))
    os.mkdir(join(script_dir, "Datasets//celeba_resized//female"))
    cg.resizetrain(train_dir, train_genders)

# Define hyper-parameters
img_size = 224
batch_size = 64
epoch = 100
lr = 0.001
split = 0.1
nodes = 512
drop = 0.25

total_training = len(os.listdir(join(script_dir, "Datasets/celeba/img")))
train_dir = join(script_dir, "Datasets/celeba_resized")
val_num = split * total_training
train_num = len(os.listdir(join(script_dir, "Datasets/celeba_resized/male"))) + \
            len(os.listdir(join(script_dir, "Datasets/celeba_resized/female"))) \
            - val_num

print("Number of training samples: ", train_num)
print("Number of validation samples: ", val_num)


def switch():
    # Training with augmentation generators
    def genTraining():
        train_generator = ImageDataGenerator(rescale=1. / 255, validation_split=split)
        train_gen = train_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(img_size, img_size),
                                                        class_mode='binary',
                                                        subset='training')

        val_gen = train_generator.flow_from_directory(batch_size=batch_size,
                                                      directory=train_dir,
                                                      target_size=(img_size, img_size),
                                                      class_mode='binary',
                                                      subset='validation')

        gender_model = cg.GenderClassify(img_size, nodes=nodes, drop=drop, normalise=False)
        opt = k.optimizers.Adam(lr)
        gender_model.compile(optimizer=opt, loss=k.losses.sparse_categorical_crossentropy, metrics=['acc'])

        # Early stopper - stops training when the program finds a local minimum
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', baseline=None, verbose=2,
                           restore_best_weights=True)

        print("Training model...")

        gender_model.fit(train_gen,
                         batch_size=batch_size,
                         epochs=epoch,
                         validation_data=val_gen,
                         verbose=1,
                         callbacks=[es])

        # Save trained model weights to local directory
        model_path = join(script_dir, "A1/Gender_classifier")
        gender_model.save(model_path, save_format='tf')
        print("Saved model to", model_path)

    # Training without augmentation generators
    def normalTrain():
        x = cg.train_arr(total_training)
        y = np.array(train_genders, dtype=np.int8)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split)

        gender_model = cg.GenderClassify(img_size, nodes=nodes, drop=drop)
        opt = k.optimizers.Adam(lr)
        gender_model.compile(optimizer=opt, loss=k.losses.sparse_categorical_crossentropy, metrics=['acc'])

        # Early stopper - stops training when the program finds a local minimum
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', baseline=None, verbose=2,
                           restore_best_weights=True)

        print("Training model...")

        gender_model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epoch,
                         validation_data=(x_val, y_val),
                         verbose=1,
                         callbacks=[es])

        # Save trained model architecture and weights to local directory
        model_path = join(script_dir, "A1/Gender_classifier")
        gender_model.save(model_path, save_format='tf')
        print("Saved model to", model_path)

    def default():
        print("Please enter a valid option.")
        switch()

    # User input
    option = int(
        input("Enter 1 for training with image augmentation\nEnter 2 for training without image augmentation:\n"))

    switch_dict = {
        1: genTraining,
        2: normalTrain,
    }

    switch_dict.get(option, default)()


switch()
