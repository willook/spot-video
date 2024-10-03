from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class GaussianBlur(AbstractImageAugmentor):
    def __init__(self, mask_size:int=5):
        self.mask_size = (mask_size, mask_size)

    def __call__(self, image):
        image = cv2.GaussianBlur(image, self.mask_size, 0)
        return image

if __name__ == '__main__':
    test = np.random.rand(100, 200, 3).astype(np.float32)
    augmentor = GaussianBlur(3)
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
