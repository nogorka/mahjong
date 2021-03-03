import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import numpy as np
import pandas as pd
import cv2
# from scipy.misc import imresize
from time import time
import os

newpath = '.augmented/'


def extract_data(path):
    data_lines = open(path+'data').readlines()
    images = []

    for line in data_lines:
        images.append(plt.imread(path+line.strip()+'.jpg'))

    return images, data_lines


def save_au(image, df, type):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    t = str(time())
    filename = type+t+'.jpg'
    plt.imsave(newpath+filename, image)
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
        rows, cols, _ = image.shape
        for r in range(rows//2, rows, rows//5):
            for c in range(cols//2, cols, cols//5):
                for offset_r, offset_c in zip(range(0, rows//2, rows//10), range(0, cols//2, cols//10)):

                    transformed = getPerspectiveTransformImage(
                        image, r, c, offset_r, offset_c)
                    augmented_df = save_au(
                        transformed, augmented_df, line.strip())

        # affine transformation
        rows, cols, _ = image.shape
        for r in range(rows//2, rows, rows//5):
            for c in range(cols//2, cols, cols//5):
                for offset_r, offset_c in zip(range(0, rows//2, rows//10), range(0, cols//2, cols//10)):

                    transformed = getAffineTransformImage(
                        image, r, c, offset_r, offset_c)
                    augmented_df = save_au(
                        transformed, augmented_df, line.strip())

    augmented_df.to_csv('.augmented/au_df', sep=';')


def getPerspectiveTransformImage(image, _row, _col, offset_r, offset_c):
    rows, cols, _ = image.shape

    pts1 = np.float32([
        [offset_c, offset_r],
        [_col+offset_c, offset_r],
        [offset_c, _row+offset_r],
        [_col, _row+offset_r]])

    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts2, pts1)

    result = cv2.warpPerspective(image, M, (cols, rows))
    return result


def getAffineTransformImage(image, _row, _col, offset_r, offset_c):
    rows, cols, _ = image.shape

    pts2 = np.float32([
        [offset_c, offset_r],
        [_col+offset_c, offset_r],
        [_col, _row+offset_r]])

    pts1 = np.float32([[0, 0], [cols, 0],  [cols, rows]])

    M = cv2.getAffineTransform(pts1, pts2)

    result = cv2.warpAffine(image, M, (cols, rows))
    return result


def getBluredImage(image):

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (7, 7), 0)

    # Median Blur
    median = cv2.medianBlur(image, 5)

    # Двустороннее размытие
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # TODO: chose and return correct blur
    return image


if __name__ == "__main__":
    form_au_df()

    # image = cv2.imread('images\prepared-set\Chun.jpg')

    # augmented_df = pd.DataFrame()
    # augmented_df[['filename', 'label']] = True

    # row, col, _ = image.shape
    # for r in range(rows//2, rows, rows//5):
    #     for c in range(cols//2, cols, cols//5):
    #         for offset_r, offset_c in zip(range(0, rows//2, rows//10), range(0, cols//2, cols//10)):

    # for r in range(row//2, row, row//5):
    #     for c in range(col//2, col, col//5):
    #         for offset_r in range(0, row//2, row//10):
    #             for offset_c in range(0, col//2, col//10):

    #                 transformed = getPerspectiveTransformImage(
    #                     image, r, c, offset_r, offset_c)
    #                 augmented_df = save_au(transformed, augmented_df, "chun")
