import cv2
import numpy as np


def convert_wandb_image(path):
    image = cv2.imread(path)
    x = image[:, :, 0]
    y = image[:, :, 1]
    z = image[:, :, 2]

    option = (z, x, y)
    image = np.dstack(option)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    return image


def display_images(images):
    for i in images:
        img = convert_wandb_image(i)
        cv2.imshow(i, img)

    cv2.waitKey(0)


if __name__ == "__main__":
    import sys

    display_images(sys.argv[1:])
