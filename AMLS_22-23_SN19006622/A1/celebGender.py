# Import packages
import pandas as pd
import splitfolders
import os

# Use PlaidML as a keras backend (Set up PlaidML using 'plaidml-setup' for GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
import keras.backend as b
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import applications as kapp
from PIL import Image, ImageOps
import shutil
import re

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Resize training images if needed
def resizetrain(train_dir, train_genders):
    for file in os.listdir(train_dir):
        f = os.path.join(train_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            im_resized = ImageOps.pad(im, (224, 224))
            if train_genders[filenum] == 0:
                im_resized.save(
                    os.path.join(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\female",
                                 file),
                    'png')
            else:
                im_resized.save(
                    os.path.join(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\male",
                                 file),
                    'png')

# Resize testing images if needed
def resizetest(test_dir, test_genders):
    for file in os.listdir(test_dir):
        f = os.path.join(test_dir, file)
        filename = os.fsdecode(file)
        filenum = int(re.findall("\d+", filename)[0])
        with Image.open(f) as im:
            im_resized = ImageOps.pad(im, (224, 224))
            if test_genders[filenum] == 0:
                im_resized.save(
                    os.path.join(
                        r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\female",
                        file),
                    'png')
            else:
                im_resized.save(
                    os.path.join(
                        r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\male",
                        file),
                    'png')


# Gender classifier model class
class GenderClassify(k.Model):

    # Default params
    img_size, batch_size, epoch, lr = 224, 100, 15, 0.0001

    def __init__(self, **kwargs):
        self.img_size = kwargs.get('')
        self.batch_size = kwargs.get('')
        self.epoch = kwargs.get('')
        self.lr = kwargs.get('')



train_labels = pd.read_csv(os.path.join(script_dir, "Datasets\celeba\labels.csv"), sep='\t')
train_genders = train_labels["gender"]
train_genders = train_genders.replace(-1, 0)

test_labels = pd.read_csv(os.path.join(script_dir, "Datasets\celeba_test\labels.csv"), sep='\t')
test_genders = test_labels["gender"]
test_genders = test_genders.replace(-1, 0)

print(train_genders.head())

train_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba\img"
test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test\img"
test_num = len(os.listdir(test_dir))

if not os.path.exists(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized"):
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\male")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\female")
    resizetrain(train_dir, train_genders)

if not os.path.exists(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"):
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\male")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\female")
    resizetest(test_dir, test_genders)


total_training = len(os.listdir(os.path.join(script_dir,"Datasets\celeba\img")))
train_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized"
test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"
val_num = 0.2*total_training
train_num = len(os.listdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\male")) + \
            len(os.listdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_resized\female"))


print("Number of training samples: ", train_num)
print("Number of validation samples: ", val_num)
print("Number of testing samples: ", test_num)
img_size = 224
batch_size = 100
epoch = 15
lr = 0.0001


image_gen_train = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(img_size, img_size),
                                                     class_mode='binary',
                                                     subset='training')

val_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                   directory=train_dir,
                                                   target_size=(img_size, img_size),
                                                   class_mode='binary',
                                                   subset='validation')

image_gen_test = ImageDataGenerator(rescale=1. / 255)
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
                                                   directory=test_dir,
                                                   target_size=(img_size, img_size),
                                                   class_mode='binary')

vgg_model = kapp.VGG16(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))

for layer in vgg_model.layers:
    #print(layer.name)
    layer.trainable = False

top_layer = vgg_model.get_layer('block5_pool')
top_output = top_layer.output
x = k.layers.GlobalMaxPool2D()(top_output)
x = k.layers.Dense(1024, activation='relu')(x)
x = k.layers.Dropout(0.5)(x)
x = k.layers.Dense(2, activation='sigmoid')(x)

gender_model = k.Model(vgg_model.input, x)
opt = k.optimizers.Adam(lr)
gender_model.compile(optimizer=opt, loss=k.losses.sparse_categorical_crossentropy, metrics=['acc'])

# Early stopper - stops training when the program finds a local minimum
es = EarlyStopping(monitor='loss', min_delta=0.0001, mode='auto', baseline=None, verbose=2)

print("Training model...")

gender_model.fit_generator(train_data_gen,
                           steps_per_epoch=(train_num // batch_size),
                           epochs=epoch,
                           validation_data=val_data_gen,
                           validation_steps=(val_num // batch_size),
                           verbose=1,
                           callbacks=[es])

# Save trained model architecture and weights to local directory
gender_model.save(os.path.join(script_dir, "A1\Gender_classifier.h5"))
gender_model.save_weights(os.path.join(script_dir, "A1\Gender_weights.h5"))
print("Saved model to disk")

print("Testing model...")
result = gender_model.evaluate_generator(test_data_gen, steps=(test_num // batch_size), verbose=1)
print("Test loss and accuracy: ", result)
