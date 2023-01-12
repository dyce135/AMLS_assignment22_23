# Import packages
import pandas as pd
import os
from os.path import join, exists

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use PlaidML as a keras backend (Set up PlaidML using 'plaidml-setup' for GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import celebGender as cg
import numpy as np

train_labels = pd.read_csv(join(script_dir, "Datasets\celeba\labels.csv"), sep='\t')
train_genders = train_labels["gender"]
train_genders = train_genders.replace(-1, 0)
print(train_genders.head())
train_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba\img"

if not exists(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized"):
    print("Resizing training data...")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\male")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\female")
    cg.resizetrain(train_dir, train_genders)

img_size = 224
batch_size = 64
epoch = 10
lr = 0.0001
split = 0.2

total_training = len(os.listdir(join(script_dir, "Datasets\celeba\img")))
train_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized"
val_num = split * total_training
train_num = len(os.listdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\male")) + \
            len(os.listdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\female")) \
            - val_num

print("Number of training samples: ", train_num)
print("Number of validation samples: ", val_num)

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

gender_model = cg.GenderClassify(img_size).model()
opt = k.optimizers.Adam(lr)
gender_model.compile(optimizer=opt, loss=k.losses.sparse_categorical_crossentropy, metrics=['acc'])

# Early stopper - stops training when the program finds a local minimum
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', baseline=None, verbose=2,
                   restore_best_weights=True)

print("Training model...")

gender_model.fit_generator(train_gen,
                           steps_per_epoch=train_num // batch_size,
                           epochs=epoch,
                           validation_data=val_gen,
                           validation_steps=val_num // batch_size,
                           verbose=1,
                           callbacks=[es])

# Save trained model architecture and weights to local directory
model_path = join(script_dir, "A1\Gender_classifier.h5")
gender_model.save(model_path)
gender_model.save_weights(join(script_dir, "A1\Gender_weights.h5"))
print("Saved model to", model_path)
