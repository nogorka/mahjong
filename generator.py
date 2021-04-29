from methods import *

NEW_PATH = ".augmented/"


class Generator(object):

    def __init__(self, h, p):
        self.helper = h
        self.path = p

    # use methods to create new image
    def generate(self):
        imgs = []
        all_labels = []
        counter = 0

        print("Start generating...")

        for img, label in zip(self.helper.images, self.helper.df):
            # work with one image (label, class)

            # not combined methods
            imgs.append(get_noised_image(img))
            imgs.append(get_blurred_image(img))

            for i in range(5):
                imgs.append(get_random_flare_image(img, 10 + i * 3))

            for i in range(5):
                imgs.append(get_resized_image(img, 95 - i * 5))

            for i in range(10):
                imgs.append(get_diff_lightness_image(img, 0.95 - i * 0.05))

            imgs += get_rotated_images(img, 10)  # second: rotation degree
            imgs += get_transformed_images(img, "Affine")
            imgs += get_transformed_images(img, "Perspective")

            # combined methods
            new_img = get_noised_image(img)
            new_img = get_blurred_image(new_img)
            for i in range(10):
                combined_img = get_diff_lightness_image(new_img, 0.95 - i * 0.05)
                imgs += get_transformed_images(combined_img, "Affine")

            new_img = get_noised_image(img)
            new_img = get_blurred_image(new_img)
            for i in range(10):
                combined_img = get_random_flare_image(new_img, 10 + i * 3)
                imgs += get_transformed_images(combined_img, "Perspective")

            # prepare labels for saving
            all_labels += [label[:-1] for _ in range(0, len(imgs))]

            # status bar
            counter += 1
            print(f"Completed label: {label[:-1]}\t-\t{counter}/{len(self.helper.df)} finished")

        print("Saving...")
        self.helper.gen_img = imgs
        self.helper.save_df(all_labels, path=self.path)
        print("Successfully finished")


if __name__ == "__main__":
    helper = Helper()
    helper.read_image("images/prepared-set/Chun", "Chun\n")
    # helper.read_frame("images/prepared-set/")

    gen = Generator(helper, NEW_PATH)
    gen.generate()
