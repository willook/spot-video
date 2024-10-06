from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt


class ReduceBitDepth(AbstractImageAugmentor):
    def __init__(self, bits=5):
        self.bits = bits

    def getName(self):
        return f"{self.__class__.__name__}({self.bits})"

    def __call__(self, image):
        # 최대값 계산
        max_value = 2**self.bits - 1

        image = np.uint8(image * 255)

        # 비트 깊이 줄이기
        image = (image // (256 // (max_value + 1))) * (256 // (max_value + 1))

        image = image.astype(np.float32) / 255

        return image


if __name__ == "__main__":
    test = np.random.rand(100, 100, 3).astype(np.float32)
    augmentor = ReduceBitDepth()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
