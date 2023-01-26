# This module contains the model architectures and functions to be used during training, testing, and cross-validation
# Import packages
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from os.path import join, exists
from PIL import Image, ImageOps
import re
import cv2 as cv

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Resize training images by class
def sort_train(train_dir, train_eyes):
    for file in os.listdir(train_dir):
        f = join(train_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize image
            im = im.resize((224, 224))
            # Save to male or female folder depending on label
            if train_eyes[filenum] == 0:
                im.save(join(join(script_dir, "Datasets//cartoon_eye//0"), file), 'png')
            elif train_eyes[filenum] == 1:
                im.save(join(join(script_dir, "Datasets//cartoon_eye//1"), file), 'png')
            elif train_eyes[filenum] == 2:
                im.save(join(join(script_dir, "Datasets//cartoon_eye//2"), file), 'png')
            elif train_eyes[filenum] == 3:
                im.save(join(join(script_dir, "Datasets//cartoon_eye//3"), file), 'png')
            elif train_eyes[filenum] == 4:
                im.save(join(join(script_dir, "Datasets//cartoon_eye//4"), file), 'png')


# Resize testing images by class
def sort_test(test_dir, test_eyes):
    for file in os.listdir(test_dir):
        f = join(test_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize image
            im = im.resize((224, 224))
            # Save to male or female folder depending on label
            if test_eyes[filenum] == 0:
                im.save(join(join(script_dir, "Datasets//cartoon_test_eye//0"), file), 'png')
            elif test_eyes[filenum] == 1:
                im.save(join(join(script_dir, "Datasets//cartoon_test_eye//1"), file), 'png')
            elif test_eyes[filenum] == 2:
                im.save(join(join(script_dir, "Datasets//cartoon_test_eye//2"), file), 'png')
            elif test_eyes[filenum] == 3:
                im.save(join(join(script_dir, "Datasets//cartoon_test_eye//3"), file), 'png')
            elif test_eyes[filenum] == 4:
                im.save(join(join(script_dir, "Datasets//cartoon_test_eye//4"), file), 'png')


# Convert training images to arrays
def train_arr(size):
    train_dat = np.empty([size, 108], dtype=np.float)
    resized_dir = join(script_dir, "Datasets//cartoon_eye")

    # Region of interest preprocessing
    def preprocessing(img_dir, file_no):
        im_dat = cv.imread(img_dir)
        im_dat = im_dat[113:122, 85:89, :]
        im_dat = np.ravel(im_dat) / 255
        train_dat[file_no] = im_dat

    for file in os.listdir(join(resized_dir, "0")):
        f = join(resized_dir, "0", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "1")):
        f = join(resized_dir, "1", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "2")):
        f = join(resized_dir, "2", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "3")):
        f = join(resized_dir, "3", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "4")):
        f = join(resized_dir, "4", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)

    return train_dat


# Convert testing images to arrays
def test_arr(size):
    test_dat = np.empty([size, 108], dtype=np.float)
    resized_dir = join(script_dir, "Datasets//cartoon_test_eye")

    # Region of interest preprocessing
    def preprocessing(img_dir, file_no):
        im_dat = cv.imread(img_dir)
        im_dat = im_dat[113:122, 85:89, :]
        im_dat = np.ravel(im_dat) / 255
        test_dat[file_no] = im_dat

    for file in os.listdir(join(resized_dir, "0")):
        f = join(resized_dir, "0", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "1")):
        f = join(resized_dir, "1", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "2")):
        f = join(resized_dir, "2", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "3")):
        f = join(resized_dir, "3", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)
    for file in os.listdir(join(resized_dir, "4")):
        f = join(resized_dir, "4", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        preprocessing(f, filenum)

    return test_dat
