from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class Gamma(AbstractImageAugmentor):
    def __init__(self, gamma: list = None):
        if gamma is None or len(gamma) != 3:
            gamma = np.random.rand(3).astype(np.float32)
            gamma = gamma * 1.0 + (1.0 - 1.0 / 2)
        self.gamma = gamma

    def getName(self):
        return f"{self.__class__.__name__}({self.gamma})"

    def __call__(self, image):
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX)

        image = [image[:, :, rgb] ** self.gamma[rgb] for rgb in range(3)]
        image = np.stack(image, axis=-1)
        return image


if __name__ == "__main__":
    test = np.random.rand(100, 100, 3).astype(np.float32)
    augmentor = Gamma()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
