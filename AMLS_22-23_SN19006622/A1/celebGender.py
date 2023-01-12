# Import packages
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy

# Use PlaidML as a keras backend (Set up PlaidML using 'plaidml-setup' for GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
from os.path import join, exists
from keras import applications as kapp
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import re
import numpy as np
import shutil
from distutils.dir_util import copy_tree

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Resize training images by class
def resizetrain(train_dir, train_genders):
    for file in os.listdir(train_dir):
        f = join(train_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize by adding black bars to images (prevents distortion)
            im_resized = ImageOps.pad(im, (224, 224))
            # Save to male or female folder depending on label
            if train_genders[filenum] == 0:
                im_resized.save(join(join(script_dir, "Datasets//celeba_resized//female"), file), 'png')
            else:
                im_resized.save(join(join(script_dir, "Datasets\celeba_resized\male"), file), 'png')


# Resize testing images by class
def resizetest(test_dir, test_genders):
    for file in os.listdir(test_dir):
        f = join(test_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            # Resize by adding black bars to images (prevents distortion)
            im_resized = ImageOps.pad(im, (224, 224))
            # Save to male or female folder depending on label
            if test_genders[filenum] == 0:
                im_resized.save(join(join(script_dir, "Datasets//celeba_test_resized//female"), file), 'png')
            else:
                im_resized.save(join(join(script_dir, "Datasets\celeba_test_resized\male"), file), 'png')


# Generate folders for validation
def valfolders(resized_dir, size):
    val_dir = join(script_dir, "Datasets//celeba_val")
    temp_dir = join(val_dir, "temp")
    os.mkdir(val_dir)
    os.mkdir(temp_dir)
    for folder in range(1, 6):
        os.mkdir(join(val_dir, str(folder)))
    copy_tree(join(resized_dir, "male"), temp_dir)
    copy_tree(join(resized_dir, "female"), temp_dir)
    index = np.arange(0, size)
    np.random.shuffle(index)
    i_split = np.split(index, 5)
    folder = 1
    for i in i_split:
        for img in i:
            image = join(temp_dir, str(img) + ".jpg")
            dest = join(val_dir, str(folder))
            shutil.copy(image, dest)
        folder += 1
    shutil.rmtree(temp_dir)


# Convert training images to arrays
def train_arr(size):
    train_dat = numpy.empty([size, 224, 224, 3])
    resized_dir = join(script_dir, "Datasets//celeba_resized")

    for file in os.listdir(join(resized_dir, "female")):
        f = join(resized_dir, "female", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            train_dat[filenum] = im_dat
    for file in os.listdir(join(resized_dir, "male")):
        f = join(resized_dir, "male", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            train_dat[filenum] = im_dat

    return train_dat


# Convert testing images to arrays
def test_arr(size):
    test_dat = numpy.empty([size, 224, 224, 3])
    resized_dir = join(script_dir, "Datasets//celeba_test_resized")

    for file in os.listdir(join(resized_dir, "female")):
        f = join(resized_dir, "female", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            test_dat[filenum] = im_dat
    for file in os.listdir(join(resized_dir, "male")):
        f = join(resized_dir, "male", file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with load_img(f) as im:
            im_dat = img_to_array(im)
            test_dat[filenum] = im_dat

    return test_dat


# Gender classifier model class
class GenderClassifyCustom(k.Model):

    def __init__(self, size, **kwargs):
        self.kernel = kwargs.get('kernel_initializer', k.initializers.RandomNormal(mean=0, stddev=0.01))
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        super().__init__()
        self.layer1 = k.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3), padding='same')
        self.layer2 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer3 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.layer4 = k.layers.Dropout(0.25)

        self.layer1 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer2 = k.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer3 = k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.layer4 = k.layers.Dropout(0.25)

        self.layer5 = k.layers.GlobalMaxPool2D()
        self.layer6 = k.layers.Dense(128, activation='relu', kernel_initializer=self.kernel,
                                     bias_initializer=self.bias)
        self.layer7 = k.layers.Dropout(0.5)
        self.layer_out = k.layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out = self.layer_out(x)
        return out

    def model(self):
        input = k.Input(shape=(224, 224, 3))
        return k.Model(input, self.call(input))


# Transfer learning gender classifier based on the pretrained VGG16 model
class GenderClassify(k.Model):

    def __init__(self, size, **kwargs):
        self.kernel = kwargs.get('kernel_initializer', k.initializers.RandomNormal(mean=0, stddev=0.01))
        self.bias = kwargs.get('kernel_initializer', k.initializers.Zeros())
        self.nodes = kwargs.get('nodes', 1024)
        super().__init__()
        self.vgg = kapp.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        for layer in self.vgg.layers:
            layer.trainable = False
        self.layer1 = k.layers.GlobalMaxPool2D()
        self.layer2 = k.layers.Dense(self.nodes, activation='relu', kernel_initializer=self.kernel,
                                     bias_initializer=self.bias)
        self.layer3 = k.layers.Dropout(0.5)
        self.layer_out = k.layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        vgg = self.vgg(inputs)
        maxpool = self.layer1(vgg)
        dense = self.layer2(maxpool)
        drop = self.layer3(dense)
        out = self.layer_out(drop)
        return out

    def model(self):
        return k.Model(self.vgg.input, self.call(self.vgg.input))
