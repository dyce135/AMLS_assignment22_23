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

    kfold = modelsel.KFold(n_splits=5, shuffle=True)

    model_acc, acc = [], []

    model_no = 1

    kernel = ['linear', 'rbf', 'poly']


    for k in kernel:
        cv_acc = []
        nk = 1

        print("\nModel ", model_no, " details: \nKernel = ", k, "\n")

        for train, val in kfold.split(x, y):
            face_model = svm.SVC(kernel=k, verbose=True)
            face_model.fit(x, y)
            print("\nValidating model ", model_no," for fold no. ", nk)
            result = face_model.score(x, y)
            cv_acc.append(result)
            nk += 1
        cv_acc = np.array(cv_acc)
        model_acc.append(np.mean(cv_acc))
        model_no += 1


    df = pd.DataFrame({'Mean accuracy': model_acc})
    df.index = ['Linear', 'RBF', 'Poly']
    print(df)
    df.to_csv(join(script_dir, "B1//face_cv.csv"))


