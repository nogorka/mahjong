import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import numpy as np
import pandas as pd
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
        for i in range(10, 360, 10):
            rotated = rotate(image, i)
            augmented_df = save_au(rotated, augmented_df, line.strip())

        # for i in range(1, 4):
        #     resized = imresize(image, i*25)
        #     save_au(resized, augmented_df, line.strip())

    augmented_df.to_csv('.augmented/au_df', sep=';')


if __name__ == "__main__":
    form_au_df()
