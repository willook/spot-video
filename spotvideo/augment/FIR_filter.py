from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class FIRFilter(AbstractImageAugmentor):
    def __init__(self, coefficient: list = None):
        if coefficient is None:
            coefficient = np.random.rand(16)

        self.coefficient = np.float32(coefficient)
        self.coefficient = self.coefficient / np.sum(self.coefficient)
        self.filter_length = len(self.coefficient)
        self.x = list()

    def getName(self):
        return f"{self.__class__.__name__}({self.coefficient})"

    def __call__(self, image):
        image = self.apply_filter(image)
        return image

    def apply_filter(self, image):
        self.x.insert(0, image)
        if len(self.x) > self.filter_length:
            self.x.pop()
        out = 0
        denominator = 0
        for coef, img in zip(self.coefficient, self.x):
            out = out + coef * img
            denominator += coef
        denominator = max(0.001, denominator)
        out /= denominator
        return out


if __name__ == "__main__":

    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    augmentor = FIRFilter()

    for idx in range(20):

        test = np.sin(idx * 0.4 + 2 * np.pi * (xx + yy))
        test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)

        output = augmentor(test)

        plt.subplot(1, 2, 1)
        plt.imshow(test, vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(output, vmin=0, vmax=1)
        plt.show()
