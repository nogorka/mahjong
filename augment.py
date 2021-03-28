import matplotlib.pyplot as plt
import scipy
from scipy.ndimage.interpolation import rotate
import numpy as np
import pandas as pd
import cv2
# from scipy.misc import imresize
from time import time
import os

newpath = '.augmented/'


def extract_data(path):
    data_lines = open(path + 'data').readlines()
    images = []

    for line in data_lines:
        images.append(cv2.imread(path + line.strip() + '.jpg'))

    return images, data_lines


def save_au(image, df, type):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    t = str(time())
    filename = type + t + '.jpg'
    plt.imsave(newpath + filename, image)
    df = df.append({'filename': filename, 'label': type}, ignore_index=True)
    return df


def form_au_df():
    path = 'images/prepared-set/'
    images, data_lines = extract_data(path)

    augmented_df = pd.DataFrame()
    augmented_df[['filename', 'label']] = True

    for image, line in zip(images, data_lines):
        # rotation
        # for i in range(10, 360, 10):
        #     rotated = rotate(image, i)
        #     augmented_df = save_au(rotated, augmented_df, line.strip())

        # size_chaging
        # for i in range(1, 4):
        #     resized = imresize(image, i*25)
        #     save_au(resized, augmented_df, line.strip())

        # perspective transformation
        # rows, cols, _ = image.shape
        # for r in range(rows//2, rows, rows//5):
        #     for c in range(cols//2, cols, cols//5):
        #         for offset_r, offset_c in zip(range(0, rows//2, rows//10), range(0, cols//2, cols//10)):
        #
        #             transformed = getPerspectiveTransformImage(
        #                 image, r, c, offset_r, offset_c)
        #             augmented_df = save_au(
        #                 transformed, augmented_df, line.strip())

        # affine transformation
        # rows, cols, _ = image.shape
        # for r in range(rows//2, rows, rows//5):
        #     for c in range(cols//2, cols, cols//5):
        #         for offset_r, offset_c in zip(range(0, rows//2, rows//10), range(0, cols//2, cols//10)):
        #
        #             transformed = getAffineTransformImage(
        #                 image, r, c, offset_r, offset_c)
        #             augmented_df = save_au(
        #                 transformed, augmented_df, line.strip())

        # poisson noise
        # noised = getNoisedImage(image)
        # augmented_df = save_au(noised, augmented_df, line.strip())

        # blur
        # blurred = getBlurredImage(image)
        # augmented_df = save_au(blurred, augmented_df, line.strip())

        # change saturation
        # another = getAnotherSaturationImage(image, 2)
        # augmented_df = save_au(another, augmented_df, line.strip())

        # random blick
        blicked = getAnotherSaturationImage(image, 2)
        augmented_df = save_au(blicked, augmented_df, line.strip())

    augmented_df.to_csv('.augmented/au_df', sep=';')


def getNoisedImage(image):
    noise = np.random.poisson(image).astype(float)
    return image + noise


def getPerspectiveTransformImage(image, _row, _col, offset_r, offset_c):
    rows, cols, _ = image.shape

    pts1 = np.float32([
        [offset_c, offset_r],
        [_col + offset_c, offset_r],
        [offset_c, _row + offset_r],
        [_col, _row + offset_r]])

    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts2, pts1)

    result = cv2.warpPerspective(image, M, (cols, rows))
    return result


def getAffineTransformImage(image, _row, _col, offset_r, offset_c):
    rows, cols, _ = image.shape

    pts2 = np.float32([
        [offset_c, offset_r],
        [_col + offset_c, offset_r],
        [_col, _row + offset_r]])

    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows]])

    M = cv2.getAffineTransform(pts1, pts2)

    result = cv2.warpAffine(image, M, (cols, rows))
    return result


def getBlurredImage(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    # median = cv2.medianBlur(image, 5)

    # Двустороннее размытие
    # bilateral = cv2.bilateralFilter(image, 9, 75, 75)

    image = convolution(gaussian)
    return gaussian


def convolution(image):
    mask = np.array([
        [-1, 2, -1],
        [2, 2, 2],
        [-1, 2, -1]
    ])

    image_bounded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_bounded[1:-1, 1:-1] = image
    result = np.zeros_like(image)
    for y in range(1, image_bounded.shape[0] - 1):
        for x in range(1, image_bounded.shape[1] - 1):
            sub = image_bounded[y - 1:y + 2, x - 1: x + 2]
            new_value = np.sum(sub * mask)  # /np.sum(mask)
            result[y - 1, x - 1] = new_value
    return result

def getAnotherSaturationImage(image, koef):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv)

    s = s * koef
    s = np.clip(s, 0, 255)

    hsv = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    return image
def getRandomBlickAtImage(image):

    return image


if __name__ == "__main__":
    # form_au_df()

    image = cv2.imread('images\prepared-set\Chun.jpg')

    augmented_df = pd.DataFrame()
    augmented_df[['filename', 'label']] = True

    # row, col, _ = image.shape
    image = getRandomBlickAtImage(image)
    augmented_df = save_au(image, augmented_df, "image")

