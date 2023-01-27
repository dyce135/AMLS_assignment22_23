# Run this file for k-fold grid search cross-validation
# Import packages
import gc
import pandas as pd
import os
import tensorflow.keras as k
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import celebSmile as cs
import sklearn.model_selection as modelsel
import numpy as np
from os.path import join, exists


def run():

    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_labels = pd.read_csv(join(script_dir, "Datasets/celeba/labels.csv"), sep='\t')
    train_smiles = train_labels["smiling"]
    train_smiles = train_smiles.replace(-1, 0)
    print(train_smiles.head())
    train_dir = join(script_dir, "Datasets/celeba/img")

    if not exists(join(script_dir, "Datasets/celeba_resized_smile")):
        print("Resizing training data...")
        os.mkdir(join(script_dir, "Datasets/celeba_resized_smile"))
        os.mkdir(join(script_dir, "Datasets/celeba_resized_smile/yes"))
        os.mkdir(join(script_dir, "Datasets/celeba_resized_smile/no"))
        cs.resizetrain(train_dir, train_smiles)

    train_num = len(os.listdir(join(script_dir, "Datasets/celeba/img")))
    train_dir = join(script_dir, "Datasets/celeba_resized_smile")

    print("Number of training samples: ", train_num)
    img_size = 224
    samples = 5000

    x = cs.train_arr(samples)
    y = np.array(train_smiles)

    kfold = modelsel.KFold(n_splits=5, shuffle=True)

    model_loss, model_loss_var, model_acc, model_acc_var = [], [], [], []

    model_no = 1

    lr = 0.0001
    stdev = [0.01, 0.05]
    nodes_list = [1024, 512]
    drop_list = [0.25, 0.5]
    epoch = 5
    batch_size = 64

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', baseline=None, verbose=2,
                       restore_best_weights=True)

    for drop in drop_list:
        for dev in stdev:
            for nodes in nodes_list:

                cv_loss, cv_acc = [], []
                nk = 1

                print("\nModel ", model_no, " details: \nKernel stddev = ", dev, "\nNodes = ", nodes, "\nDropout = ", drop, "\n")

                for train, val in kfold.split(x, y):
                    smile_model = cs.SmileClassify(img_size, drop=drop, nodes=nodes, normalise=True, kernel_stddev=dev)
                    opt = k.optimizers.Adam(lr)
                    smile_model.compile(optimizer=opt, loss=k.losses.binary_crossentropy, metrics=['acc'])
                    print("Training model ", model_no, " for fold no. ", nk)
                    smile_model.fit(x[train], y[train],
                                    batch_size=batch_size,
                                    epochs=epoch,
                                    verbose=1)
                    print("Validating model for fold no. ", nk)
                    result = smile_model.evaluate(x[val], y[val], batch_size=batch_size, verbose=1)
                    cv_loss.append(result[0])
                    cv_acc.append(result[1])
                    nk += 1
                    del smile_model
                    K.clear_session()
                    gc.collect()
                cv_loss = np.array(cv_loss)
                cv_acc = np.array(cv_acc)
                model_loss.append(np.mean(cv_loss))
                model_acc.append(np.mean(cv_acc))
                model_no += 1

    df = pd.DataFrame({'Mean accuracy': model_acc, 'Mean Loss': model_loss})
    df.index = ['stddev = 0.01, Nodes = 1024, Dropout = 0.25', 'stddev = 0.01, Nodes = 512, Dropout = 0.25', 'stddev = 0.05, Nodes = 1024, Dropout = 0.25',
                'stddev = 0.05, Nodes = 512, Dropout = 0.25', 'stddev = 0.01, Nodes = 1024, Dropout = 0.5', 'stddev = 0.01, Nodes = 512, Dropout = 0.5', 'stddev = 0.05, Nodes = 1024, Dropout = 0.5',
                'stddev = 0.05, Nodes = 512, Dropout = 0.5']
    print(df)
    df.to_csv(join(script_dir, "A2/smile_cv.csv"))

