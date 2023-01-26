import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skimage import feature
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from os.path import join, exists
from keras import applications as kapp
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import re

size = 224
samples = 5000
splits = 4
points = 8
rad = 2
eps = 1e-7

train_dat = np.empty([samples, (splits ** 2) * (points + 2)])
train_dir = "/Datasets/celeba_resized"

for file in os.listdir(join(train_dir, "female")):
    tiles = np.empty([splits ** 2, points + 2])
    f = join(join(train_dir, "female"), file)
    filename = os.fsdecode(file)
    filenum = int(re.findall("\d+", filename)[0])
    img = load_img(f, 'L')
    im_dat = np.array(img)
    k = 0
    for i in range(0, size, size // splits):
        for j in range(0, size, size // splits):
            temp = im_dat[i:i + size // splits, j:j + size // splits]
            lbp = feature.local_binary_pattern(temp, points, rad, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            tiles[k, :] = hist
            k += 1

    tiles = tiles.flatten()
    train_dat[filenum] = tiles

for file in os.listdir(join(train_dir, "male")):
    tiles = np.empty([splits ** 2, points + 2])
    f = join(join(train_dir, "male"), file)
    filename = os.fsdecode(file)
    filenum = int(re.findall("\d+", filename)[0])
    img = load_img(f, 'L')
    im_dat = np.array(img)
    k = 0
    for i in range(0, size, size // splits):
        for j in range(0, size, size // splits):
            temp = im_dat[i:i + size // splits, j:j + size // splits]
            lbp = feature.local_binary_pattern(temp, points, rad, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            tiles[k, :] = hist
            k += 1

    tiles = tiles.flatten()
    train_dat[filenum] = tiles


quit()
