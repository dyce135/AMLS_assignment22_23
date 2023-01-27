# Run this file for k-fold grid search cross-validation
# Import packages
import gc
import pandas as pd
import os
import cartoonFaces as cf
import sklearn.model_selection as modelsel
from sklearn import svm
import numpy as np
from os.path import join, exists


def run():
    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_labels = pd.read_csv(join(script_dir, "Datasets/cartoon_set/labels.csv"), sep='\t')
    train_faces = train_labels["face_shape"]
    train_faces = train_faces.replace(-1, 0)
    print(train_faces.head())
    train_dir = join(script_dir, "Datasets/cartoon_set/img")

    if not exists(join(script_dir, "Datasets/cartoon_face")):
        print("Resizing training data...")
        os.mkdir(join(script_dir, "Datasets/cartoon_face"))
        os.mkdir(join(script_dir, "Datasets//cartoon_face//0"))
        os.mkdir(join(script_dir, "Datasets//cartoon_face//1"))
        os.mkdir(join(script_dir, "Datasets//cartoon_face//2"))
        os.mkdir(join(script_dir, "Datasets//cartoon_face//3"))
        os.mkdir(join(script_dir, "Datasets//cartoon_face//4"))
        cf.sort_train(train_dir, train_faces)

    train_num = len(os.listdir(join(script_dir, "Datasets/cartoon_set/img")))
    train_dir = join(script_dir, "Datasets/cartoon_face")

    print("Number of training samples: ", train_num)
    img_size = 224
    samples = 10000

    x = cf.train_arr(samples)
    y = np.array(train_faces)

    # Run repeated k-fold and find significance
    cv = modelsel.KFold(n_splits=5, shuffle=True)

    model_acc = []

    linear_model = svm.SVC(kernel='linear')
    rbf_model = svm.SVC(kernel='rbf')

    linear_score = modelsel.cross_val_score(linear_model, x, y, scoring='accuracy', n_jobs=-1, cv=cv)
    rbf_score = modelsel.cross_val_score(rbf_model, x, y, scoring='accuracy', n_jobs=-1, cv=cv)

    lin = np.array(linear_score)
    rbf = np.array(rbf_score)

    model_acc.append(lin.mean())
    model_acc.append(rbf.mean())

    df = pd.DataFrame({'Mean accuracy': model_acc})
    df.index = ['Linear', 'RBF']
    print(df)
    df.to_csv(join(script_dir, "B1//face_cv.csv"))






