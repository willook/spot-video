from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class PerspectiveTransform(AbstractImageAugmentor):
    def __init__(self, matrix:np.array=None):
        if matrix is None:
            mag = np.random.rand(1)[0] * 5 + 4
            ref = np.float32([[mag, mag], [mag, -mag], [-mag, mag], [-mag, -mag]])
            points1 = np.random.rand(4, 2).astype(np.float32)
            points1 += ref
            points2 = np.random.rand(4, 2).astype(np.float32)
            points2 += ref
            self.updateMatrixWithPoints(points1, points2)

    def __call__(self, image):
        height, width, _ = image.shape
        image = cv2.warpPerspective(image, self.matrix, (width, height))
        return image
    
    def updateMatrixWithPoints(self, orig_points, transformed_points):
        orig_points = np.float32(orig_points)
        transformed_points = np.float32(transformed_points)
        self.matrix = cv2.getPerspectiveTransform(orig_points, transformed_points)

if __name__ == '__main__':
    test = np.random.rand(100, 200, 3).astype(np.float32)
    augmentor = PerspectiveTransform()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
