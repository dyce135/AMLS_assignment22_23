# Run this file for model training
# Import packages
import pandas as pd
import os
from os.path import join, exists
from sklearn import svm
import cartoonEyes as cf
import numpy as np
import pickle as pk


def run():
    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Read labels
    train_labels = pd.read_csv(join(script_dir, "Datasets/cartoon_set/labels.csv"), sep='\t')
    train_eyes = train_labels["eye_color"]
    print(train_eyes.head())
    train_dir = join(script_dir, "Datasets/cartoon_set/img")

    # Resize and reorganise images to classes
    if not exists(join(script_dir, "Datasets/cartoon_eye")):
        print("Sorting training data...")
        os.mkdir(join(script_dir, "Datasets/cartoon_eye"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//0"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//1"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//2"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//3"))
        os.mkdir(join(script_dir, "Datasets//cartoon_eye//4"))
        cf.sort_train(train_dir, train_eyes)

    train_num = len(os.listdir(join(script_dir, "Datasets/cartoon_set/img")))
    train_dir = join(script_dir, "Datasets/cartoon_resized_eye")

    print("Number of training samples: ", train_num)

    x = cf.train_arr(train_num)
    y = np.array(train_eyes)

    eye_model = svm.SVC(kernel='linear', verbose=True)
    eye_model.fit(x, y)
    model_dir = join(script_dir, 'B2/eye_classifier.sav')
    pk.dump(eye_model, open(model_dir, 'wb'))

    print('SVM classifier saved to ', model_dir)
