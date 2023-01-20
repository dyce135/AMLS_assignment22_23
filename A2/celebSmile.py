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


# smile classifier model class
class SmileClassifyCustom(k.Model):

    # Constructor with layers
    def __init__(self, size, **kwargs):
        self.kernel = kwargs.get('kernel_initializer', k.initializers.RandomNormal(mean=0, stddev=0.01))
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        self.nodes = kwargs.get('nodes', 1024)
        self.drop = kwargs.get('drop', 0.5)
        self.normalise = kwargs.get('normalise', True)
        super().__init__()
        self.rescale = k.layers.Rescaling(1./255)

        self.layer1 = k.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3), padding='same')
        self.layer2 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer3 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.layer4 = k.layers.Dropout(0.25)

        self.layer1 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer2 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer3 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.layer4 = k.layers.Dropout(0.25)

        self.layer5 = k.layers.GlobalMaxPool2D()
        self.layer6 = k.layers.Dense(self.nodes, activation='relu', kernel_initializer=self.kernel,
                                     bias_initializer=self.bias)
        self.layer7 = k.layers.Dropout(self.drop)
        self.layer_out = k.layers.Dense(2, activation='sigmoid')

    # Call function to connect layers
    def call(self, inputs):
        if self.normalise:
            x = self.rescale(inputs)
        else:
            x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out = self.layer_out(x)
        return out


# Transfer learning smile classifier based on the pretrained VGG16 model
class SmileClassify(k.Model):

    # Constructor with layers
    def __init__(self, size, **kwargs):
        self.kernel = kwargs.get('kernel_initializer', k.initializers.RandomNormal(mean=0, stddev=0.01))
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
        self.layer_out = k.layers.Dense(2, activation='sigmoid')

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