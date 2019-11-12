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
    return img_1 / np.linalg.norm(img_1), img_2 / np.linalg.norm(img_2)

def load_chars_dataset():
    """
    loads the chars74k-lite dataset
    returns:
    X : np.array -> images
    y : np.array -> numeric labels
    """
    char_to_int = init_char_to_int()

    dir = "dataset/chars74k-lite"
    X = []
    y = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                img = Image.open(root + "/" + file)
                img.load()
                img = np.asarray(img)

                #feature engineering. invert colors of image to expand data set
                img_invert = np.copy(img)
                img_invert = np.invert(img_invert)

                #feature engineering. Removing noise from the images
                img = denoise_tv_chambolle(img, weight=0.1, multichannel=False)
                img_invert = denoise_tv_chambolle(img_invert, weight=0.1, multichannel=False)

                #feature engineering. Normalizing the image vectors
                img = img / np.linalg.norm(img)
                img_invert = img_invert / np.linalg.norm(img_invert)

                #flatten the images into 2-d arrays
                img = img.flatten()
                img_invert = img_invert.flatten()

                label = char_to_int[root[-1]]

                X.append(img)
                y.append(label)

                X.append(img_invert)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

def show_image(x):
    plt.imshow(np.reshape(x, (20, 20)), cmap="gray")
