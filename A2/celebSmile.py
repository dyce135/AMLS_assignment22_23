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
from skimage import feature

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Resize training images by class
def resizetrain(train_dir, train_smiles):
    for file in os.listdir(train_dir):
        f = join(train_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize by adding black bars to images (prevents distortion)
            im_resized = ImageOps.pad(im, (224, 224))
            # Save to male or female folder depending on label
            if train_smiles[filenum] == 0:
                im_resized.save(join(join(script_dir, "Datasets/celeba_resized_smile/no"), file), 'png')
            else:
                im_resized.save(join(join(script_dir, "Datasets/celeba_resized_smile/yes"), file), 'png')


# Resize testing images by class
def resizetest(test_dir, test_smiles):
    for file in os.listdir(test_dir):
        f = join(test_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize by adding black bars to images (prevents distortion)
            im_resized = ImageOps.pad(im, (224, 224))
            # Save to male or female folder depending on label
            if test_smiles[filenum] == 0:
                im_resized.save(join(join(script_dir, "Datasets/celeba_test_resized_smile/no"), file), 'png')
            else:
                im_resized.save(join(join(script_dir, "Datasets/celeba_test_resized_smile/yes"), file), 'png')


# Convert training images to arrays
def train_arr(size):
    train_dat = np.empty([size, 224, 224, 3], dtype=np.uint8)
    resized_dir = join(script_dir, "Datasets/celeba_resized_smile")

    for file in os.listdir(join(resized_dir, "no")):
        f = join(resized_dir, "no", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            train_dat[filenum] = im_dat
    for file in os.listdir(join(resized_dir, "yes")):
        f = join(resized_dir, "yes", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            train_dat[filenum] = im_dat

    return train_dat


# Convert testing images to arrays
def test_arr(size):
    test_dat = np.empty([size, 224, 224, 3], dtype=np.uint8)
    resized_dir = join(script_dir, "Datasets/celeba_test_resized_smile")

    for file in os.listdir(join(resized_dir, "no")):
        f = join(resized_dir, "no", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            test_dat[filenum] = im_dat
    for file in os.listdir(join(resized_dir, "yes")):
        f = join(resized_dir, "yes", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            test_dat[filenum] = im_dat

    return test_dat

# Generate histogram data for training
# Method adapted from the web article "Local Binary Patterns with Python and OpenCV" by Adrian Rosebrock
# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def train_hist(samples, splits, points, rad, eps, size):
    train_dat = np.empty([samples, (splits ** 2) * (points + 2)])
    train_dir = join(script_dir, "Datasets/celeba_resized_smile")

    # Find feature vector for each image
    for file in os.listdir(join(train_dir, "yes")):
        tiles = np.empty([splits ** 2, points + 2])
        f = join(join(train_dir, "yes"), file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        img = load_img(f, 'L')
        im_dat = np.array(img)
        k = 0
        # Split image into splits^2 sub-images
        for i in range(0, size, size // splits):
            for j in range(0, size, size // splits):
                # Find feature vector from concatenated histograms
                temp = im_dat[i:i + size // splits, j:j + size // splits]
                lbp = feature.local_binary_pattern(temp, points, rad, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)
                tiles[k, :] = hist
                k += 1

        tiles = tiles.flatten()
        train_dat[filenum] = tiles

    for file in os.listdir(join(train_dir, "no")):
        tiles = np.empty([splits ** 2, points + 2])
        f = join(join(train_dir, "no"), file)
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

    return train_dat


# Generate histogram data for testing
# Method adapted from the web article "Local Binary Patterns with Python and OpenCV" by Adrian Rosebrock
# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def test_hist(samples, splits, points, rad, eps, size):
    train_dat = np.empty([samples, (splits ** 2) * (points + 2)])
    train_dir = join(script_dir, "Datasets/celeba_test_resized_smile")

    for file in os.listdir(join(train_dir, "yes")):
        tiles = np.empty([splits ** 2, points + 2])
        f = join(join(train_dir, "yes"), file)
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

    for file in os.listdir(join(train_dir, "no")):
        tiles = np.empty([splits ** 2, points + 2])
        f = join(join(train_dir, "no"), file)
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

    return train_dat


# smile classifier model class
class SmileClassifyLBP(k.Model):

    # Constructor with layers
    def __init__(self, size, **kwargs):
        self.kernel = kwargs.get('kernel_initializer', k.initializers.RandomNormal(mean=0, stddev=0.01))
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        self.inp = kwargs.get('input_shape', (160,))
        super().__init__()
        self.layer1 = k.layers.Conv1D(128, activation='relu', kernel_size=3, input_shape=self.inp)
        self.layer2 = k.layers.Dense(512, activation='relu', kernel_initializer=self.kernel,
                                     bias_initializer=self.bias)
        self.layer_conv = k.layers.Conv1D(128, activation='relu', kernel_size=3)
        self.layer_pool = k.layers.MaxPooling1D(pool_size=2, strides=2)
        self.layer3 = k.layers.Flatten()
        self.layer_out = k.layers.Dense(1, activation='sigmoid')

    # Call function to connect layers
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer_out(x)
        return out

# Transfer learning smile classifier based on the pretrained VGG16 model
class SmileClassify(k.Model):

    # Constructor with layers
    def __init__(self, size, **kwargs):
        self.kernel_dev = kwargs.get('kernel_stddev', 0.01)
        self.kernel = k.initializers.RandomNormal(mean=0, stddev=self.kernel_dev)
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        self.nodes = kwargs.get('nodes', 1024)
        self.drop = kwargs.get('drop', 0.5)
        self.normalise = kwargs.get('normalise', True)
        super().__init__()
        self.rescale = k.layers.Rescaling(1./255)
        self.vgg = kapp.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        for layer in self.vgg.layers:
            layer.trainable = False
        if self.normalise:
            self.vgg_new = k.Model(self.vgg.layers[1].input, self.vgg.output)
        self.layer1 = k.layers.GlobalMaxPool2D()
        self.layer2 = k.layers.Dense(self.nodes, activation='relu', kernel_initializer=self.kernel,
                                     bias_initializer=self.bias)
        self.layer3 = k.layers.Dropout(self.drop)
        self.layer_out = k.layers.Dense(1, activation='sigmoid')

    # Call function to connect layers
    def call(self, inputs):
        if self.normalise is True:
            rescale = self.rescale(inputs)
            vgg = self.vgg(rescale)
        else:
            vgg = self.vgg(inputs)
        maxpool = self.layer1(vgg)
        dense = self.layer2(maxpool)
        drop = self.layer3(dense)
        out = self.layer_out(drop)
        return out