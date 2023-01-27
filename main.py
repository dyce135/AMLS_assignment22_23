from os import system as run
from os.path import join
import sys

# Import relevant files
sys.path.insert(0, 'A1')
import celebGender_CV
import celebGender_train
import celebGender_test
sys.path.insert(0, 'A2')
import celebSmile_CV
import celebSmile_train
import celebSmile_test
sys.path.insert(0, 'B1')
import cartoonFaces_CV
import cartoonFaces_train
import cartoonFaces_test
sys.path.insert(0, 'B2')
import cartoonEyes_CV
import cartoonEyes_train
import cartoonEyes_test

# Task A1 switch
def switch_gender():
    def gcv():
        celebGender_CV.run()

    def gtrain():
        celebGender_train.run()

    def gtest():
        celebGender_test.run()

    def gdefault():
        print("Please enter a valid option.")
        switch_gender()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: gcv,
        2: gtrain,
        3: gtest,
    }

    switch_dict.get(option, gdefault)()


# Task A2 switch
def switch_smile():

    def scv():
        celebSmile_CV.run()

    def strain():
        celebSmile_train.run()

    def stest():
        celebSmile_test.run()

    def sdefault():
        print("Please enter a valid option.")
        switch_smile()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: scv,
        2: strain,
        3: stest,
    }

    switch_dict.get(option, sdefault)()


# Task B1 switch
def switch_face():

    def fcv():
        cartoonFaces_CV.run()

    def ftrain():
        cartoonFaces_CV.run()

    def ftest():
        cartoonFaces_CV.run()

    def fdefault():
        print("Please enter a valid option.")
        switch_face()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: fcv,
        2: ftrain,
        3: ftest,
    }

    switch_dict.get(option, fdefault)()


# Task B2 switch
def switch_eye():

    def ecv():
        cartoonEyes_CV.run()

    def etrain():
        cartoonEyes_train.run()

    def etest():
        cartoonEyes_test.run()

    def edefault():
        print("Please enter a valid option.")
        switch_eye()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: ecv,
        2: etrain,
        3: etest,
    }

    switch_dict.get(option, edefault)()


# Main switch
def switch_main():
    # Task A1
    def gender():
        switch_gender()

    # Task A2
    def smile():
        switch_smile()

    # Task B1
    def face():
        switch_face()

    # Task B2
    def eye():
        switch_eye()

    def default():
        print("Please enter a valid option.")
        switch_main()

    def quit_main():
        quit()

    # User input
    option = int(
        input(
            "Enter 1 for task A1\nEnter 2 for task A2\nEnter 3 for task B1\nEnter 4 for task B2\n"))

    switch_dict = {
        1: gender,
        2: smile,
        3: face,
        4: eye,
        5: quit_main,
    }

    switch_dict.get(option, default)()

# Call switch
switch_main()
quit()
