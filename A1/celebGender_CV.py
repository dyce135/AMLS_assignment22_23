__author__ = "Konrad Chan"

# Run this file for k-fold grid search cross-validation
# Import packages
import pandas as pd
import os

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import celebGender as cg
import sklearn.model_selection as modelsel
import numpy as np
from os.path import join, exists
import gc

train_labels = pd.read_csv(join(script_dir, "Datasets/celeba/labels.csv"), sep='\t')
train_genders = train_labels["gender"]
train_genders = train_genders.replace(-1, 0)
print(train_genders.head())
train_dir = join(script_dir, "Datasets/celeba/img")

if not exists(join(script_dir, "Datasets/celeba_resized")):
    print("Resizing training data...")
    os.mkdir(join(script_dir, "Datasets/celeba_resized"))
    os.mkdir(join(script_dir, "Datasets/celeba_resized/male"))
    os.mkdir(join(script_dir, "Datasets/celeba_resized/female"))
    cg.resizetrain(train_dir, train_genders)

train_num = len(os.listdir(join(script_dir, "Datasets/celeba/img")))
train_dir = join(script_dir, "Datasets/celeba_resized")

print("Number of training samples: ", train_num)
img_size = 224
samples = 5000

x = cg.train_arr(samples)
y = np.array(train_genders, dtype=np.int8)

kfold = modelsel.KFold(n_splits=5, shuffle=True)

model_loss, model_acc, loss, acc = [], [], [], []

model_no = 1

learning_rate = [0.0001, 0.001]
nodes_list = [512, 1024]
drop_list = [0.25, 0.5]
epoch = 4
batch_size = 64

for drop in drop_list:
    for lr in learning_rate:
        for nodes in nodes_list:

            cv_loss, cv_acc = [], []
            nk = 1

            print("\nModel ", model_no, " details: \nLR = ", lr, "\nNodes = ", nodes, "\nDropout = ", drop, "\n")

            for train, val in kfold.split(x, y):
                gender_model = cg.GenderClassify(img_size, drop=drop, nodes=nodes, normalise=True)
                opt = tf.keras.optimizers.Adam(lr)
                gender_model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy,
                                     metrics=['acc'])
                print("Training model ", model_no, " for fold no. ", nk)
                gender_model.fit(x[train], y[train],
                                 batch_size=batch_size,
                                 epochs=epoch,
                                 validation_data=(x[val], y[val]),
                                 verbose=1)
                print("Validating model for fold no. ", nk)
                result = gender_model.evaluate(x[val], y[val], batch_size=batch_size, verbose=1)
                cv_loss.append(result[0])
                cv_acc.append(result[1])
                nk += 1
                del gender_model
                K.clear_session()
                gc.collect()
            cv_loss = np.array(cv_loss)
            cv_acc = np.array(cv_acc)
            model_loss.append(np.mean(cv_loss))
            model_acc.append(np.mean(cv_acc))
            model_no += 1

df = pd.DataFrame({'Mean accuracy': model_acc, 'Mean Loss': model_loss})
df.index = ['LR = 0.0001, Nodes = 1024, Dropout = 0.25', 'LR = 0.0001, Nodes = 512, Dropout = 0.25',
            'LR = 0.001, Nodes = 1024, Dropout = 0.25',
            'LR = 0.001, Nodes = 512, Dropout = 0.25', 'LR = 0.0001, Nodes = 1024, Dropout = 0.5',
            'LR = 0.0001, Nodes = 512, Dropout = 0.5', 'LR = 0.001, Nodes = 1024, Dropout = 0.5',
            'LR = 0.001, Nodes = 512, Dropout = 0.5']
print(df)
df.to_csv(join(script_dir, "A1//gender_cv.csv"))
