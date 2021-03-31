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

        return self.images, self.df

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


if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun")

    helper.read_frame("images/prepared-set/")

    helper.save(helper.images[0], "Chun", NEW_PATH)
