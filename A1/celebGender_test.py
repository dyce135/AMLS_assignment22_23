# Import packages
import pandas as pd
import os

# Use PlaidML as a keras backend (Set up PlaidML using 'plaidml-setup' for GPU acceleration)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras as k
from keras.preprocessing.image import ImageDataGenerator
import celebGender as cg

# Project path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


test_labels = pd.read_csv(os.path.join(script_dir, "Datasets\celeba_test\labels.csv"), sep='\t')
test_genders = test_labels["gender"]
test_genders = test_genders.replace(-1, 0)

test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test\img"
test_num = len(os.listdir(test_dir))

if not os.path.exists(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"):
    print("Resizing testing data...")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\male")
    os.mkdir(r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized\female")
    cg.resizetest(test_dir, test_genders)

test_dir = r"C:\Users\konra\PycharmProjects\AMLS_22-23_SN19006622\Datasets\celeba_test_resized"
print("Number of testing samples: ", test_num)
img_size = 224
batch_size = 64

image_gen_test = ImageDataGenerator(rescale=1. / 255)
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
                                                   directory=test_dir,
                                                   target_size=(img_size, img_size),
                                                   class_mode='binary')
x_test, y_test = test_data_gen.next()

model_dir = os.path.join(script_dir, "A1\Gender_classifier.h5")

model = k.models.load_model(model_dir)

print("Testing model...")
result = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("Test loss and accuracy: ", result)
