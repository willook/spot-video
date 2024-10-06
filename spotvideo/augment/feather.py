from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class Feather(AbstractImageAugmentor):
    def __init__(self, distance=100):
        self.distance = distance
        self.weight = None

    def getName(self):
        return f"{self.__class__.__name__}({self.distance})"

    def __call__(self, image):
        if self.weight is None:
            self.update_weight(image)

        if self.weight.shape != image.shape:
            self.update_weight(image)

        out = image * self.weight
        return out

    def update_weight(self, image):
        ones = np.zeros(image.shape[0:-1], dtype=np.uint8)
        ones[1:-1, 1:-1] = 1

        self.weight = cv2.distanceTransform(ones, cv2.DIST_L2, 3)
        self.weight = np.clip(self.weight, a_min=0, a_max=self.distance)
        self.weight = cv2.normalize(self.weight, None, 0, 1.0, cv2.NORM_MINMAX)
        self.weight = np.stack([self.weight, self.weight, self.weight], axis=-1)


if __name__ == "__main__":
    test = np.random.rand(1024, 1024, 3).astype(np.float32)
    augmentor = Feather()

    output = augmentor(test)

    plt.subplot(1, 3, 1)
    plt.imshow(test)
    plt.subplot(1, 3, 2)
    plt.imshow(augmentor.weight)
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.show()
