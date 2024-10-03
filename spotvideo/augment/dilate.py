from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class Dilate(AbstractImageAugmentor):
    def __init__(self, mask_size:int=5):
        self.kernel = np.ones((mask_size, mask_size), dtype=np.uint8)

    def __call__(self, image):
        image = cv2.dilate(image, self.kernel, iterations=1)
        return image

if __name__ == '__main__':
    test = np.random.rand(1024, 1024, 3).astype(np.float32)
    augmentor = Dilate(3)
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
