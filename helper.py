import matplotlib.pyplot as plt
import pandas as pd
import cv2
from time import time
import os

NEW_PATH = ".augmented/"


class Helper(object):

    def __init__(self):
        self.images = []  # loaded images
        self.df = []  # loaded dataframe with marked data

        self.gen_img = []  # generated images
        self.gen_df = pd.DataFrame()
        self.gen_df[["filename", "label"]] = True

    # read images and data file
    def read_frame(self, path):
        self.df = open(path + "data").readlines()

        for line in self.df:
            self.images.append(cv2.imread(path + line.strip() + ".jpg"))

    # read one image for testing purposes 
    def read_image(self, path, name):
        self.images.append(cv2.imread(path + ".jpg"))
        self.df.append(name)

    # save generated image and record to dataframe file
    def save(self, img, label, path):
        if not os.path.exists(path):
            os.makedirs(path)

        filename = label + str(time()) + ".jpg"

        plt.imsave(path + filename, img)
        self.gen_df = self.gen_df.append({"filename": filename, "label": label}, ignore_index=True)

        self.gen_df.to_csv(r'.augmented\data.csv', sep=';', index=False)

    # save generated images and dataframe
    def save_df(self, labels, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for img, label in zip(self.gen_img, labels):
            filename = label + str(time()) + ".jpg"

            plt.imsave(path + filename, img)
            self.gen_df = self.gen_df.append({"filename": filename, "label": label}, ignore_index=True)

        self.gen_df.to_csv(r'.augmented\data.csv', sep=';', index=False)

    # refresh generated data
    def refresh(self):
        self.gen_df = []
        self.gen_img = []


if __name__ == "__main__":
    helper = Helper()

    helper.read_image("images/prepared-set/Chun", "Chun")

    print(helper.df[0])
    plt.imshow(helper.images[0])
    plt.show()

    # helper.read_frame("images/prepared-set/")

    # for i in range(len(helper.images)):
    #     print(helper.df[i])
    #     plt.imshow(helper.images[i])
    #     plt.show()

    helper.save(helper.images[0], "Chun", NEW_PATH)
