# AMLS Assignment 22/23: SN19006622

Project repository for the 2022-2023 ELEC00134: Applied Machine Learning Systems coursework. In part A, we exlore the use of convolutional neural networks (CNNs) as well as preprocessing techniques such as local binary patterns and data augmentation in the training on the CelebA dataset for binary classificationo. In part B, we explore the use of support vector machines (SVMs) along with preprocessing techniques for feature extraction such as canny edges extraction and region of interest analysis in the training on the Cartoon Set dataset.

## Environment and dependent packages

This project was developed in Python 3.10.9 using the following packages and the following specified versions:
- tensorflow-rocm 2.11.0 (Alternatively, use **tensorflow-gpu 2.11.0** or **tensorflow 2.11.0**)
- keras 2.11.0
- skimage 0.19.3
- numpy 1.22.4
- pillow 9.3.0
- pandas 1.5.3
- sklearn 1.2.0
- matplotlib 3.6.2
- opencv 4.6.0
- pickle 3.1.0

Using other versions of the same packages may result in incompatibilities or errors.

## Datasets

The models are built for the CelebA dataset by S. Yang et al. (S. Yang, P. Luo, C. C. Loy, and X. Tang, "From facial parts responses to face detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015) in Part A, and the Cartoon Set dataset by google (https://google.github.io/cartoonset/) in Part B.

## Structure of this project

To run this project, ensure that the celeba, celeba_test, cartoon_set, and cartoon_set_test folders are in the /Datasets directory, then simply run the main file and select the appropriate options to begin training and testing. Each folder contains a main file to store functions and models (e.g. celebGender.py), and also separate files to run the training, cross-validation, and testing for these models. The folders also contain files to store trained models and weights, as well as informational files such as graphs, plots, and tables to be viewed at your discretion. Files to test and demonstrate image preprocessing techniques are also included in some folders.
