import matplotlib.pyplot as plt

from helper import Helper
from methods import Methods

NEW_PATH = ".augmented/"


class Generator(object):

    def __init__(self, helper):
        self.helper = helper
        self.methods = Methods()

    def generate(self):
        for image, line in zip(self.helper.images, self.helper.df):
            # work with one image

            blicked = self.methods.get_random_flare_image(image)
            plt.imshow(blicked)
            plt.show()
            helper.save(blicked, line, NEW_PATH)

        helper.gen_df.to_csv('.augmented/gen_df', sep=';')


def gen():
    pass
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
    # blicked = getNoisedImage(image)
    # plt.imshow(blicked)
    # plt.show()
    # helper.gen_df = helper.save(blicked, line.strip())



if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun")
    # helper.read_frame("images/prepared-set/")

    gen = Generator(helper)
    gen.generate()