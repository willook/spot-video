from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class MultiplyNoise(AbstractImageAugmentor):
    def __init__(self, mean=1, sigma=0.3):
        self.mean = mean
        self.sigma = sigma
        # Generate Gaussian noise
        self.gaussian_noise = None

    def getName(self):
        return f"{self.__class__.__name__}({self.mean}, {self.sigma})"

    def __call__(self, image):
        if self.gaussian_noise is None:
            self.updateNoise(image)
        image = cv2.multiply(image, self.gaussian_noise)
        image = np.clip(image, a_min=0, a_max=1)

        return image

    def updateNoise(self, image):
        self.gaussian_noise = np.random.normal(
            self.mean, self.sigma, image.shape
        ).astype(np.float32)


if __name__ == "__main__":
    test = np.random.rand(1000, 1000, 3).astype(np.float32)
    augmentor = MultiplyNoise()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
