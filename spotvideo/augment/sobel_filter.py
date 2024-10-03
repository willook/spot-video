from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class SobelFilter(AbstractImageAugmentor):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, image):
        # Apply Sobel filter
        # Sobel in X direction
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=self.kernel_size)

        # Sobel in Y direction
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=self.kernel_size)

        # Combine the results (magnitude of the gradient)
        image = np.sqrt(sobel_x**2 + sobel_y**2)

        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX)

        return image


if __name__ == '__main__':
    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    test = np.sin(2 * np.pi * (xx + yy))

    test = np.stack([test, test, test], axis=-1).astype(np.float32)
    test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)

    augmentor = SobelFilter()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
