# This module contains the model architectures and functions to be used during training, testing, and cross-validation
# Import packages
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow.keras as k
from os.path import join, exists
from keras import applications as kapp
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import re
import cv2 as cv

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resized_dir = join(script_dir, "Datasets/cartoon_test_resized_face/0/11.png")

im_dat = cv.imread(resized_dir, 0)
cv.imshow('image',im_dat)
canny = cv.Canny(im_dat, 224, 224)
canny = canny[30:194, 50:174]
cv.imshow('canny edges',canny)
cv.waitKey(0)

quit()
