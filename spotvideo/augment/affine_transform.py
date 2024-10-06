from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class AffineTransform(AbstractImageAugmentor):
    def __init__(self, matrix: np.array = None):
        self.matrix = matrix
        if matrix is None:
            mag = np.random.rand() * 5 + 4
            offset = np.random.rand() * 100 - 50
            ref = np.float32([[mag, mag], [mag, -mag], [-mag, -mag]])
            points1 = np.random.rand(3, 2).astype(np.float32)
            points1 += ref
            points2 = np.random.rand(3, 2).astype(np.float32)
            points2 += ref + offset
            self.updateMatrixWithPoints(points1, points2)

    def getName(self):
        return f"{self.__class__.__name__}({self.matrix})"

    def __call__(self, image):
        height, width, _ = image.shape
        image = cv2.warpAffine(image, self.matrix, (width, height))
        return image

    def updateMatrixWithPoints(self, orig_points, transformed_points):
        orig_points = np.float32(orig_points)
        transformed_points = np.float32(transformed_points)
        self.matrix = cv2.getAffineTransform(orig_points, transformed_points)


if __name__ == "__main__":
    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    test = np.sin(1.5 * np.pi * (xx + yy))

    test = np.stack([test, test, test], axis=-1).astype(np.float32)
    test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)
    augmentor = AffineTransform()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
