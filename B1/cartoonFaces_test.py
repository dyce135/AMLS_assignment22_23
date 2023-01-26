# Run this file for model testing
# Import packages
import pandas as pd
import os

import numpy as np
import pickle
import cartoonFaces as cf
from os.path import join, exists
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

test_labels = pd.read_csv(os.path.join(script_dir, "Datasets/cartoon_set_test/labels.csv"), sep='\t')
test_faces = test_labels["face_shape"]

test_dir = join(script_dir, "Datasets/cartoon_set_test/img")
test_num = len(os.listdir(test_dir))

if not os.path.exists(join(script_dir, "Datasets/cartoon_test_face")):
    print("Resizing testing data...")
    os.mkdir(join(script_dir, "Datasets/cartoon_test_face"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_face//0"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_face//1"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_face//2"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_face//3"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_face//4"))
    cf.sort_test(test_dir, test_faces)

test_dir = join(script_dir, "Datasets/cartoon_test_face")
print("Number of testing samples: ", test_num)

x = cf.test_arr(test_num)
y = np.array(test_faces)

model_dir = join(script_dir, 'B1/face_classifier.sav')
model = pickle.load(open(model_dir, 'rb'))

result = model.score(x, y)
print("Test accuracy: ", result)
y_pred = model.predict(x)
y_pred = np.rint(y_pred)
confusion = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()
plt.show()

