from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class FrameDrop(AbstractImageAugmentor):
    def __init__(self, rate:int=None):
        if rate is None:
            rate = 4
        self.rate = rate
        self.count = 0
        self.out_image = None

    def __call__(self, image):
        if self.count % self.rate == 0:
            self.count = 0
            self.out_image = image
            
        self.count += 1
        return self.out_image


if __name__ == '__main__':

    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    augmentor = FrameDrop()

    for idx in range(20):

        test = np.sin(idx * 0.4 + 2 * np.pi * (xx + yy))
        test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)
        
        output = augmentor(test)

        plt.subplot(1, 2, 1)
        plt.imshow(test, vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(output, vmin=0, vmax=1)
        plt.show()
