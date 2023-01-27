# Run this file for model testing
# Import packages
import pandas as pd
import os
import keras as k
from keras.preprocessing.image import ImageDataGenerator
import celebGender as cg
from os.path import join, exists
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def run():

    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_labels = pd.read_csv(os.path.join(script_dir, "Datasets/celeba_test/labels.csv"), sep='\t')
    test_genders = test_labels["gender"]
    test_genders = test_genders.replace(-1, 0)

    test_dir = join(script_dir, "Datasets/celeba_test/img")
    test_num = len(os.listdir(test_dir))

    if not os.path.exists(join(script_dir, "Datasets/celeba_test_resized")):
        print("Resizing testing data...")
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized"))
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized/male"))
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized/female"))
        cg.resizetest(test_dir, test_genders)

    test_dir = join(script_dir, "Datasets/celeba_test_resized")
    print("Number of testing samples: ", test_num)
    img_size = 224
    batch_size = 64


    def switch():
        # Test with generators
        def gen_test():
            test_generator = ImageDataGenerator(rescale=1. / 255)
            test_gen = test_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=test_dir,
                                                          target_size=(img_size, img_size),
                                                          class_mode='binary')

            gender_model = k.models.load_model(join(script_dir, "A1/Gender_classifier_gen"))

            print("Testing model...")
            result = gender_model.evaluate(test_gen, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

        # Test without generators
        def normal_test():
            x = cg.test_arr(test_num)
            y = np.array(test_genders, dtype=np.int8)

            gender_model = k.models.load_model(join(script_dir, "A1/Gender_classifier"))

            print("Testing model...")
            result = gender_model.evaluate(x, y, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

            y_pred = gender_model.predict(x, batch_size=batch_size, verbose=1)
            y_pred = np.rint(y_pred)
            confusion = confusion_matrix(y, y_pred, normalize='pred')
            disp = ConfusionMatrixDisplay(confusion, display_labels=['Female', 'Male'])
            disp.plot(values_format='.5g')
            plt.savefig(join(script_dir, 'A1/confusion'))
            plt.show()
            print(confusion)

        # Test with lbp model
        def lbp_test():
            x = cg.test_hist(1000, 4, 8, 2, 1e-7, 224)
            y = np.array(test_genders, dtype=np.int8)
            x = x.reshape((1000, 160, 1))

            gender_model = k.models.load_model(join(script_dir, "A1/Gender_classifier_lbp"))

            print("Testing model...")
            result = gender_model.evaluate(x, y, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

            y_pred = gender_model.predict(x, batch_size=batch_size, verbose=1)
            y_pred = np.rint(y_pred)
            confusion = confusion_matrix(y, y_pred, normalize='pred')
            disp = ConfusionMatrixDisplay(confusion, display_labels=['Female', 'Male'])
            disp.plot(values_format='.5g')
            plt.savefig(join(script_dir, 'A1/confusion_lbp'))
            plt.show()
            print(confusion)

        def default():
            print("Please enter a valid option.")
            switch()

        # User input
        option = int(
            input("Enter 1 for testing with image augmentation\nEnter 2 for testing without image augmentation\nEnter 3 for testing using local binary patterns:\n"))

        switch_dict = {
            1: gen_test,
            2: normal_test,
            3: lbp_test,
        }

        switch_dict.get(option, default)()


    switch()

