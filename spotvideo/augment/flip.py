from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
from enum import Enum

from matplotlib import pyplot as plt

class FlipDirection(Enum):
    Vertical = 0
    Horizontal = 1
    Diagonal = 2

class Flip(AbstractImageAugmentor):
    def __init__(self, direction : FlipDirection = None):
        if direction is None:
            direction = np.random.choice(FlipDirection)
        self.direction = direction

    def __call__(self, image):
        if self.direction is FlipDirection.Horizontal:
            return np.flip(image, axis=1)
        if self.direction is FlipDirection.Vertical:
            return np.flip(image, axis=0)
        if self.direction is FlipDirection.Diagonal:
            return np.flip(image, axis=[0, 1])

if __name__ == '__main__':
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x, y)

    test = xx + yy

    test = np.stack([test, test, test], axis=-1).astype(np.float32)

    augmentor = Flip(FlipDirection.Diagonal)

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
