# Import packages
import pandas as pd
import splitfolders
import os

# Use PlaidML as a keras backend (AMD GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import re

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Resize images if needed
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


test_labels = pd.read_csv(os.path.join(script_dir, "Datasets\celeba_test\labels.csv"), sep='\t')
test_genders = test_labels["gender"]
test_genders = test_genders.replace(-1, 0)

test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test\img"
test_num = len(os.listdir(test_dir))

if not os.path.exists(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"):
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\male")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\female")
    resizetest(test_dir, test_genders)

test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"
print("Number of testing samples: ", test_num)
img_size = 224
batch_size = 100
epoch = 2

image_gen_test = ImageDataGenerator(rescale=1. / 255)
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
                                                   directory=test_dir,
                                                   target_size=(img_size, img_size),
                                                   class_mode='binary')

model_dir = os.path.join(script_dir, "A1\Gender_classifier.h5")
model = k.models.load_model(model_dir)

print("Testing model...")
result = model.evaluate_generator(test_data_gen, steps=(test_num // batch_size), verbose=1)
print("Test loss and accuracy: ", result)
