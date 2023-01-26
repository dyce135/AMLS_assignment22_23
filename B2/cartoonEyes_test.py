# Run this file for model testing
# Import packages
import pandas as pd
import os

import numpy as np
import pickle
import cartoonEyes as cf
from os.path import join, exists
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

test_labels = pd.read_csv(os.path.join(script_dir, "Datasets/cartoon_set_test/labels.csv"), sep='\t')
test_eyes = test_labels["eye_color"]

test_dir = join(script_dir, "Datasets/cartoon_set_test/img")
test_num = len(os.listdir(test_dir))

if not os.path.exists(join(script_dir, "Datasets/cartoon_test_eye")):
    print("Sorting testing data...")
    os.mkdir(join(script_dir, "Datasets/cartoon_test_eye"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_eye//0"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_eye//1"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_eye//2"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_eye//3"))
    os.mkdir(join(script_dir, "Datasets//cartoon_test_eye//4"))
    cf.sort_test(test_dir, test_eyes)

test_dir = join(script_dir, "Datasets/cartoon_test_eye")
print("Number of testing samples: ", test_num)

x = cf.test_arr(test_num)
y = np.array(test_eyes)

model_dir = join(script_dir, 'B2/eye_classifier.sav')
model = pickle.load(open(model_dir, 'rb'))

result = model.score(x, y)
print("Test accuracy: ", result)
y_pred = model.predict(x)
y_pred = np.rint(y_pred)
confusion = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()
plt.show()

