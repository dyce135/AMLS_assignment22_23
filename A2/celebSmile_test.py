# Run this file for model testing
# Import packages
import pandas as pd
import os
import numpy as np
import tensorflow.keras as k
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import celebSmile as cs
from os.path import join, exists
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def run():

    # Project path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_labels = pd.read_csv(os.path.join(script_dir, "Datasets/celeba_test/labels.csv"), sep='\t')
    test_smiles = test_labels["smiling"]
    test_smiles = test_smiles.replace(-1, 0)

    test_dir = join(script_dir, "Datasets/celeba_test/img")
    test_num = len(os.listdir(test_dir))

    if not os.path.exists(join(script_dir, "Datasets/celeba_test_resized_smile")):
        print("Resizing testing data...")
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized_smile"))
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized_smile/yes"))
        os.mkdir(join(script_dir, "Datasets/celeba_test_resized_smile/no"))
        cs.resizetest(test_dir, test_smiles)

    test_dir = join(script_dir, "Datasets/celeba_test_resized_smile")
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

            smile_model = k.models.load_model(join(script_dir, "A2/smile_classifier"))

            print("Testing model...")
            result = smile_model.evaluate(test_gen, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

        # Test without generators
        def normal_test():
            x = cs.test_arr(test_num)
            y = np.array(test_smiles, dtype=np.int8)

            smile_model = k.models.load_model(join(script_dir, "A2/smile_classifier"))

            print("Testing model...")
            result = smile_model.evaluate(x, y, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

            y_pred = smile_model.predict(x, batch_size=batch_size, verbose=1)
            y_pred = np.rint(y_pred)
            confusion = confusion_matrix(y, y_pred, normalize='pred')
            disp = ConfusionMatrixDisplay(confusion, display_labels=['Not smiling', 'Smiling'])
            disp.plot(values_format='.5g')
            plt.savefig(join(script_dir, 'A2/confusion'))
            plt.show()
            print(confusion)

        # Test with lbp model
        def lbp_test():
            x = cs.test_hist(1000, 4, 8, 2, 1e-7, 224)
            y = np.array(test_smiles, dtype=np.int8)
            x = x.reshape((1000, 160, 1))

            model = k.models.load_model(join(script_dir, "A2/Smile_classifier_lbp"))

            print("Testing model...")
            result = model.evaluate(x, y, batch_size=batch_size, verbose=1)
            print("Test loss and accuracy: ", result)

            y_pred = model.predict(x, batch_size=batch_size, verbose=1)
            y_pred = np.rint(y_pred)
            confusion = confusion_matrix(y, y_pred, normalize='pred')
            disp = ConfusionMatrixDisplay(confusion, display_labels=['Not smiling', 'Smiling'])
            disp.plot(values_format='.5g')
            plt.savefig(join(script_dir, 'A2/confusion_lbp'))
            plt.show()
            print(confusion)


        def default():
            print("Please enter a valid option.")
            switch()

        # User input
        option = int(
            input("Enter 1 for testing with image augmentation\nEnter 2 for training without image augmentation\nEnter 3 for testing using local binary patterns:\n"))

        switch_dict = {
            1: gen_test,
            2: normal_test,
            3: lbp_test,
        }

        switch_dict.get(option, default)()


    switch()

