from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class AddNoise(AbstractImageAugmentor):
    def __init__(self, mean=0, sigma=0.1):
        self.mean = mean
        self.sigma = sigma

    def getName(self):
        return f"{self.__class__.__name__}({self.mean}, {self.sigma})"

    def __call__(self, image):
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(self.mean, self.sigma, image.shape).astype(
            np.float32
        )
        # poisson_noise = np.random.poisson(image * 255).astype(np.float32) / 255.0
        # poisson_noise -= np.mean(poisson_noise)

        # Add the Gaussian noise to the image
        image = cv2.add(image, gaussian_noise)
        # image = cv2.add(image, poisson_noise)
        image = np.clip(image, a_min=0, a_max=1)

        return image


if __name__ == "__main__":
    test = np.random.rand(100, 100, 3).astype(np.float32)
    augmentor = AddNoise()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
