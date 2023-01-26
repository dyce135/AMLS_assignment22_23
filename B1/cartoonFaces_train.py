# Run this file for model training
# Import packages
import pandas as pd
import os
from os.path import join, exists

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn import svm
import cartoonFaces as cf
import numpy as np
import pickle as pk

# Read labels
train_labels = pd.read_csv(join(script_dir, "Datasets/cartoon_set/labels.csv"), sep='\t')
train_faces = train_labels["face_shape"]
print(train_faces.head())
train_dir = join(script_dir, "Datasets/cartoon_set/img")

# Resize and reorganise images to classes
if not exists(join(script_dir, "Datasets/cartoon_face")):
    print("Sorting training data...")
    os.mkdir(join(script_dir, "Datasets/cartoon_face"))
    os.mkdir(join(script_dir, "Datasets//cartoon_face//0"))
    os.mkdir(join(script_dir, "Datasets//cartoon_face//1"))
    os.mkdir(join(script_dir, "Datasets//cartoon_face//2"))
    os.mkdir(join(script_dir, "Datasets//cartoon_face//3"))
    os.mkdir(join(script_dir, "Datasets//cartoon_face//4"))
    cf.sort_train(train_dir, train_faces)

train_num = len(os.listdir(join(script_dir, "Datasets/cartoon_set/img")))
train_dir = join(script_dir, "Datasets/cartoon_resized_face")

print("Number of training samples: ", train_num)

x = cf.train_arr(train_num)
y = np.array(train_faces)

face_model = svm.SVC(kernel='linear', verbose=True)
face_model.fit(x, y)
model_dir = join(script_dir, 'B1/face_classifier.sav')
pk.dump(face_model, open(model_dir, 'wb'))

print('SVM classifier saved to ', model_dir)
