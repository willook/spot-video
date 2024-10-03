from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class Identity(AbstractImageAugmentor):
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.copy()
        return image

if __name__ == '__main__':
    test = np.random.rand(100, 100, 3).astype(np.float32)
    augmentor = Identity()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
