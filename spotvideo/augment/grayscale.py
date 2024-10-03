from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class GrayScale(AbstractImageAugmentor):
    def __init__(self):
        pass

    def __call__(self, image):
        out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        out = np.stack([out, out, out], axis=-1)
        return out

if __name__ == '__main__':
    test = np.random.rand(5, 5, 3).astype(np.float32)
    augmentor = GrayScale()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
