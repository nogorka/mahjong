import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from helper import Helper
from scipy.ndimage import convolve

NEW_PATH = ".augmented/"


class Methods(object):

    def get_random_flare_image(self, img, area):
        # area - one side in pixels

        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, 0]

        max_y, max_x = np.shape(img)[0], np.shape(img)[1]
        rand_x = int(random.random() * max_x)
        rand_y = int(random.random() * max_y)
        rand_coord = [rand_x, rand_y]

        mask = np.zeros_like(img[:, :]).astype('float32')

        # make random points in AREA with value 0->1
        mask = self.get_with_points(mask, rand_coord, area // 2)
        mask *= 5

        # round circle
        kernel = self.get_round_kernel(area)

        mask = convolve(mask, kernel)
        mask = cv2.GaussianBlur(mask, (area * 2 + 1, area * 2 + 1), 0)

        new_image = self.add_flare(img, mask)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)

        return new_image

    @staticmethod
    def get_with_points(img, center, area):
        start_x = center[0] - area if center[0] - area > 0 else 0
        end_x = center[0] + area if center[0] + area < img.shape[1] else img.shape[1]

        start_y = center[1] - area if center[1] - area > 0 else 0
        end_y = center[1] + area if center[1] + area < img.shape[0] else img.shape[0]

        img[start_y:end_y, start_x:end_x] = np.random.random((area * 2, area * 2))

        return img

    def get_round_kernel(self, size):
        r = int(size // 2)

        quarter = np.zeros((r, r))
        for y in range(quarter.shape[0]):
            for x in range(quarter.shape[1]):
                if x ** 2 + y ** 2 < r ** 2:
                    quarter[y][x] = 1

        half = np.concatenate([quarter[::-1], quarter], axis=0)
        k = np.concatenate([half[..., ::-1], half], axis=1)
        return k

    def add_flare(self, img, flare):
        image = np.copy(img)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if flare[y, x] > 0:
                    if image[y, x] + flare[y, x] > 255:
                        image[y, x] = 255
                    else:
                        image[y, x] += flare[y, x]

                    print(image[y, x], flare[y, x])

        plt.imshow(image)

        return image


if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun")
    image = helper.images[0]

    methods = Methods()
    image = methods.get_random_flare_image(image, 21)
    helper.save(image, "Chun", NEW_PATH)

    #
    # @staticmethod
    # def get_noised_image(image):
    #     noise = np.random.poisson(image).astype(float)
    #     return image + noise
    #
    # @staticmethod
    # def get_perspective_image(image, _row, _col, offset_r, offset_c):
    #     rows, cols, _ = image.shape
    #
    #     pts1 = np.float32([
    #         [offset_c, offset_r],
    #         [_col + offset_c, offset_r],
    #         [offset_c, _row + offset_r],
    #         [_col, _row + offset_r]])
    #
    #     pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    #
    #     M = cv2.getPerspectiveTransform(pts2, pts1)
    #
    #     result = cv2.warpPerspective(image, M, (cols, rows))
    #     return result
    #
    # @staticmethod
    # def get_affine_image(image, _row, _col, offset_r, offset_c):
    #     rows, cols, _ = image.shape
    #
    #     pts2 = np.float32([
    #         [offset_c, offset_r],
    #         [_col + offset_c, offset_r],
    #         [_col, _row + offset_r]])
    #
    #     pts1 = np.float32([[0, 0], [cols, 0], [cols, rows]])
    #
    #     M = cv2.getAffineTransform(pts1, pts2)
    #
    #     result = cv2.warpAffine(image, M, (cols, rows))
    #     return result
    #
    # @staticmethod
    # def convolution(image):
    #     mask = np.array([
    #         [-1, 2, -1],
    #         [2, 2, 2],
    #         [-1, 2, -1]
    #     ])
    #
    #     image_bounded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    #     image_bounded[1:-1, 1:-1] = image
    #     result = np.zeros_like(image)
    #     for y in range(1, image_bounded.shape[0] - 1):
    #         for x in range(1, image_bounded.shape[1] - 1):
    #             sub = image_bounded[y - 1:y + 2, x - 1: x + 2]
    #             new_value = np.sum(sub * mask)  # /np.sum(mask)
    #             result[y - 1, x - 1] = new_value
    #     return result
    #
    # @staticmethod
    # def get_blurred_image(self, image):
    #     gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    #
    #     # median = cv2.medianBlur(image, 5)
    #
    #     # Двустороннее размытие
    #     # bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    #
    #     image = self.convolution(gaussian)
    #     return gaussian
    #
    # @staticmethod
    # def get_another_saturation_image(image, koef):
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    #     (h, s, v) = cv2.split(hsv)
    #
    #     s = s * koef
    #     s = np.clip(s, 0, 255)
    #
    #     hsv = cv2.merge([h, s, v])
    #     image = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    #
    #     return image
    #
