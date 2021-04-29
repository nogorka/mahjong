import matplotlib.pyplot as plt

from helper import Helper
from methods import *

NEW_PATH = ".augmented/"


class Generator(object):

    def __init__(self, helper):
        self.helper = helper

    # use methods to create new image
    def generate(self):
        images = []
        labels = []
        counter = 0

        print("Start generating...")

        for img, label in zip(self.helper.images, self.helper.df):
            # work with one image (label, class)

            # not combined methods
            images.append(get_noised_image(img))
            images.append(get_blurred_image(img))

            for i in range(5):
                images.append(get_random_flare_image(img, 20 + i * 2))

            for i in range(5):
                images.append(get_resized_image(img, 95 - i * 5))

            for i in range(10):
                images.append(get_diff_lightness_image(img, 0.95 - i * 0.05))

            images += get_rotated_images(img, 10)  # second: rotation degree
            images += get_transformed_images(img, "Affine")
            images += get_transformed_images(img, "Perspective")

            # combined methods

            # TODO

            # prepare labels for saving
            labels += [label[:-1] for i in range(0, len(images))]

            # status bar
            counter += 1
            print(f"Completed label: {label[:-1]} - {counter}/{len(self.helper.df)} finished")
            if counter == 10:
                break

        helper.gen_img = images
        helper.save_df(labels, path=NEW_PATH)
        print("Successfully finished")


if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun")
    # helper.read_frame("images/prepared-set/")

    gen = Generator(helper)
    gen.generate()
