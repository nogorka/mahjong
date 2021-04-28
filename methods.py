import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from helper import Helper
from scipy.ndimage import convolve
from scipy.ndimage.interpolation import rotate

NEW_PATH = ".augmented/"


# 1
def get_random_flare_image(img, area):
    # area - one side in pixels

    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, 0]

    # find center coordinates
    max_y, max_x = np.shape(img)[0], np.shape(img)[1]
    rand_x = int(random.random() * max_x)
    rand_y = int(random.random() * max_y)
    rand_coord = [rand_x, rand_y]

    mask = np.zeros_like(img[:, :]).astype('float32')

    # make random points in AREA with value 0->1
    mask = image_with_points(mask, rand_coord, area // 2)
    mask *= 5

    # round circle
    kernel = round_kernel(area)

    mask = convolve(mask, kernel)
    mask = cv2.GaussianBlur(mask, (area * 2 + 1, area * 2 + 1), 0)

    new_image = add_flare(img, mask)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)

    return new_image


def image_with_points(img, center, area):
    start_x = center[0] - area if center[0] - area > 0 else 0
    end_x = center[0] + area if center[0] + area < img.shape[1] else img.shape[1]

    start_y = center[1] - area if center[1] - area > 0 else 0
    end_y = center[1] + area if center[1] + area < img.shape[0] else img.shape[0]

    img[start_y:end_y, start_x:end_x] = np.random.random((area * 2, area * 2))

    return img


def round_kernel(size):
    r = int(size // 2)

    quarter = np.zeros((r, r))
    for y in range(quarter.shape[0]):
        for x in range(quarter.shape[1]):
            if x ** 2 + y ** 2 < r ** 2:
                quarter[y][x] = 1

    half = np.concatenate([quarter[::-1], quarter], axis=0)
    k = np.concatenate([half[..., ::-1], half], axis=1)
    return k


def add_flare(img, flare):
    im = np.copy(img)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if flare[y, x] > 0:
                if im[y, x] + flare[y, x] > 255:
                    im[y, x] = 255
                else:
                    im[y, x] += flare[y, x]

                # print(im[y, x], flare[y, x])

    plt.imshow(im)

    return im


# 2
def get_noised_image(img):
    return np.random.poisson(img)


# 3
def get_diff_lightness_image(img, value):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    (h, l, s) = cv2.split(hls)

    l = np.round(l * value, decimals=0)
    l = np.clip(l, 0, 255).astype('uint8')

    hls = cv2.merge([h, l, s])

    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


# 4
def get_blurred_image(img):
    return cv2.GaussianBlur(np.copy(img), (5, 5), 0)


def perspective_image(img, _row, _col, offset_r, offset_c):
    rows, cols, _ = img.shape

    pts1 = np.float32([
        [offset_c, offset_r],
        [_col + offset_c, offset_r],
        [offset_c, _row + offset_r],
        [_col, _row + offset_r]])

    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts2, pts1)

    result = cv2.warpPerspective(img, M, (cols, rows))
    return result


def affine_image(img, _row, _col, offset_r, offset_c):
    rows, cols, _ = img.shape

    pts2 = np.float32([
        [offset_c, offset_r],
        [_col + offset_c, offset_r],
        [_col, _row + offset_r]])

    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows]])

    M = cv2.getAffineTransform(pts1, pts2)

    result = cv2.warpAffine(img, M, (cols, rows))
    return result


# 5
def get_transformed_images(img, mode="None"):
    result = []

    rows, cols, _ = img.shape
    for r in range(rows // 2, rows, rows // 5):
        for c in range(cols // 2, cols, cols // 5):

            # distortion
            for offset_r, offset_c in zip(range(0, rows // 4, rows // 10), range(0, cols // 4, cols // 10)):

                if mode == "Affine":
                    result.append(
                        affine_image(img, r, c, offset_r, offset_c))
                    result.append(
                        affine_image(rotate(img, 180), r, c, offset_r, offset_c))

                if mode == "Perspective":
                    result.append(
                        perspective_image(img, r, c, offset_r, offset_c))
                    result.append(
                        perspective_image(rotate(img, 180), r, c, offset_r, offset_c))
    return result


# 6
def get_resized_image(img, scale_percent):
    im = np.copy(img)

    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)

    return cv2.resize(im, (width, height))


# 7
def get_rotated_images(img, step):
    rotated = []
    for i in range(10, 360, step):
        rotated.append(rotate(img, i))
    return rotated


if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun")

    image = helper.images[0]

    plt.imshow(image)
    plt.show()

    # image = get_random_flare_image(image, 20)
    # image = get_noised_image(image)
    # image = get_resized_image(image, 25)
    # image = get_blurred_image(image)

    image = get_diff_lightness_image(image, 0.7)

    plt.imshow(image)
    plt.show()

    # images = []
    # images = get_rotated_images(image, 10) # second: rotation degree
    # images = get_transformed_images(image, "Affine")
    # images = get_transformed_images(image, "Perspective")

    # helper.gen_img = images
    # labels = ["Chun" for i in range(0, len(images))]

    # helper.save_df(labels, path=NEW_PATH)
