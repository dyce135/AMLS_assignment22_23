# Run this file for k-fold grid search cross-validation
# Import packages
import gc
import pandas as pd
import os
import cartoonEyes as cf
import sklearn.model_selection as modelsel
from sklearn import svm
import numpy as np
from os.path import join, exists
from scipy.stats import ttest_ind as ttest


def run():
    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_labels = pd.read_csv(join(script_dir, "Datasets/cartoon_set/labels.csv"), sep='\t')
    train_eyes = train_labels["eye_color"]
    print(train_eyes.head())
    train_dir = join(script_dir, "Datasets/cartoon_set/img")

    # Sort images
    if not exists(join(script_dir, "Datasets/cartoon_eye")):
        print("Resizing training data...")
        os.mkdir(join(script_dir, "Datasets/cartoon_eye"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//0"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//1"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//2"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//3"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//4"))
        cf.sort_train(train_dir, train_eyes)

    train_num = len(os.listdir(join(script_dir, "Datasets/cartoon_set/img")))
    train_dir = join(script_dir, "Datasets/cartoon_eye")

    print("Number of training samples: ", train_num)
    samples = 10000

    x = cf.train_arr(samples)
    y = np.array(train_eyes)

    # Run repeated k-fold and find significance
    cv = modelsel.RepeatedKFold(n_splits=5, n_repeats=20)

    model_acc, model_sd = [], []

    linear_model = svm.SVC(kernel='linear')
    rbf_model = svm.SVC(kernel='rbf')

    linear_score = modelsel.cross_val_score(linear_model, x, y, scoring='accuracy', n_jobs=-1, cv=cv)
    rbf_score = modelsel.cross_val_score(rbf_model, x, y, scoring='accuracy', n_jobs=-1, cv=cv)

    lin = np.array(linear_score)
    rbf = np.array(rbf_score)

    ttest_score = ttest(rbf, lin)

    model_acc.append(lin.mean())
    model_acc.append(rbf.mean())

    model_sd.append(lin.std())
    model_sd.append(rbf.std())

    df = pd.DataFrame({'Mean accuracy': model_acc, 'Accuracy standard deviation': model_sd})
    df.index = ['Linear', 'RBF']
    print(df)
    df.to_csv(join(script_dir, "B2//eye_cv.csv"))

    df2 = pd.DataFrame({'T-test': ttest_score})
    df2.index = ['T statistic', 'P value']
    df2.to_csv(join(script_dir, "B2//ttest.csv"))

    print("Independent t-test scores: ", ttest_score)


