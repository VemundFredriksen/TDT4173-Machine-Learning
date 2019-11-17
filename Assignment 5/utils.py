import os
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage import feature
from sklearn.preprocessing import binarize

letters = "abcdefghijklmnopqrstuvwxyz"


def init_char_to_int():
    """
    creates mapping between letters and corresponding
    ordered integer
    """
    label = 0
    char_to_int = {}
    for i in range(len(letters)):
        char_to_int[letters[i]] = label
        label += 1
    return char_to_int

def load_detection_images():
    """
    load images to do image detection on
    """
    dir = "dataset/detection-images/detection-images/"
    img_1 = Image.open(dir + "detection-1.jpg")
    img_2 = Image.open(dir + "detection-2.jpg")
    img_1 = np.asarray(img_1)
    img_2 = np.asarray(img_2)
    return img_1, img_2

def load_chars_dataset():
    """
    loads the chars74k-lite dataset
    returns:
    X : np.array -> images
    y : np.array -> numeric labels
    """
    char_to_int = init_char_to_int()

    dir = "dataset/chars74k-lite"
    #training set
    X = []
    #labels
    y = []
    #image representation of training set
    images = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                img = Image.open(root + "/" + file)
                img.load()
                img = np.asarray(img)

                #feature engineering. invert colors of image to expand data set
                img_invert = np.copy(img)
                img_invert = np.invert(img_invert)


                #feature engineering. Normalizing the image vectors
                img = img / np.linalg.norm(img)
                img_invert = img_invert / np.linalg.norm(img_invert)

                #feature engineering. Turn images into hog representations.
                img, img_representation = feature.hog(img, pixels_per_cell=(2,2), cells_per_block=(1,1), visualize=True)
                img_invert, img_representation_invert = feature.hog(img_invert, pixels_per_cell=(2,2), cells_per_block=(1,1), visualize=True)


                #flatten the images into 2-d arrays
                img = img.flatten()
                img_invert = img_invert.flatten()

                label = char_to_int[root[-1]]

                #add data to correct datasets
                X.append(img)
                y.append(label)

                X.append(img_invert)
                y.append(label)

                images.append(img_representation)
                images.append(img_representation_invert)

    X = np.array(X)
    y = np.array(y)
    images = np.array(images)

    return X, y, images

def get_label(i):
    """
    convert int label to correct letter
    """
    return letters[i]

def show_image(x):
    """
    just prints flattened imgs as 20x20 images
    """
    plt.imshow(np.reshape(x, (20, 20)), cmap="gray")
