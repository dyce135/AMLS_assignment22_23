# Import packages
import pandas as pd
import os

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use PlaidML as a keras backend (Set up PlaidML using 'plaidml-setup' for GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import celebGender as cg
import sklearn.model_selection as modelsel
import numpy as np
from os.path import join, exists

train_labels = pd.read_csv(join(script_dir, "Datasets\celeba\labels.csv"), sep='\t')
train_genders = train_labels["gender"]
train_genders = train_genders.replace(-1, 0)
print(train_genders.head())
train_dir = join(script_dir, "Datasets\celeba\img")

if not exists(join(script_dir, "Datasets\celeba_resized")):
    print("Resizing training data...")
    os.mkdir(join(script_dir, "Datasets\celeba_resized"))
    os.mkdir(join(script_dir, "Datasets\celeba_resized\male"))
    os.mkdir(join(script_dir, "Datasets//celeba_resized//female"))
    cg.resizetrain(train_dir, train_genders)

train_num = len(os.listdir(join(script_dir, "Datasets\celeba\img")))
train_dir = join(script_dir, "Datasets\celeba_resized")

print("Number of training samples: ", train_num)
img_size = 224
batch_size = 64
samples = 5000
epoch = 1

x = cg.train_arr(samples)/255
y = np.array(train_genders)

kfold = modelsel.KFold(n_splits=5, shuffle=True)

nk = 1

model_loss, model_acc = [], []
model_no = 1

learning_rate = [0.0001, 0.001]
nodes_list = [1024, 512]

for lr in learning_rate:
    for nodes in nodes_list:
        k.backend.clear_session()
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', baseline=None, verbose=2,
                           restore_best_weights=True)
        gender_model = cg.GenderClassify(img_size).model()
        opt = k.optimizers.Adam(lr)
        gender_model.compile(optimizer=opt, loss=k.losses.sparse_categorical_crossentropy, metrics=['acc'])
        cv_loss, cv_acc = [], []

        for train, val in kfold.split(x, y):
            print("Training model ", model_no, " for fold no. ", nk)
            gender_model.fit(x[train], y[train],
                             batch_size=batch_size,
                             epochs=epoch,
                             validation_data=(x[val], y[val]),
                             verbose=1,
                             callbacks=[es])
            print("validating model for fold no. ", nk)
            result = gender_model.evaluate(x[val], y[val], steps=(len(x[val]) // batch_size), verbose=1)
            cv_loss.append(result[0])
            cv_acc.append(result[1])
            nk += 1
        cv_loss = np.array(cv_loss)
        cv_acc = np.array(cv_acc)
        model_loss.append(np.mean(cv_loss))
        model_acc.append(np.mean(cv_acc))
        model_no += 1

print(model_loss, model_acc)