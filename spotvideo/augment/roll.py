from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2
from enum import Enum

from matplotlib import pyplot as plt

class RollDirection(Enum):
    Vertical = 0
    Horizontal = 1

class Roll(AbstractImageAugmentor):
    # y = ax + b
    def __init__(self, offset_ratio:float=None, direction:RollDirection=None):
        if offset_ratio is None:
            offset_ratio = np.random.rand()
        if direction is None:
            direction = np.random.choice(RollDirection)

        self.offset_ratio = offset_ratio
        self.direction = direction

    def __call__(self, image):
        if self.direction is RollDirection.Horizontal:
            return np.roll(image, int(self.offset_ratio * image.shape[1]), axis=1)
        if self.direction is RollDirection.Vertical:
            return np.roll(image, int(self.offset_ratio * image.shape[0]), axis=0)
        raise NotImplementedError

if __name__ == '__main__':
    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    test = np.sin(1.5 * np.pi * (xx + yy))

    test = np.stack([test, test, test], axis=-1).astype(np.float32)
    test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)

    augmentor = Roll()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
