from .abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from .affine_transform import AffineTransform
from .identity import Identity

from matplotlib import pyplot as plt


class ZeroingEdge(AbstractImageAugmentor):
    def __init__(self, zeroing_ratio: list = None, fit_to_crop: bool = None):
        if zeroing_ratio is None or len(zeroing_ratio) != 4:
            zeroing_ratio = np.clip(np.random.rand(4) - 0.6, a_min=0, a_max=0.4)
        self.zeroing_ratio = zeroing_ratio

        if fit_to_crop is None:
            fit_to_crop = True if np.random.rand() > 0.6 else False
        self.fit_to_crop = fit_to_crop
        if self.fit_to_crop:
            self.warp = AffineTransform()
        else:
            self.warp = Identity()

        self.shape = None

    def getName(self):
        return f"{self.__class__.__name__}({self.zeroing_ratio}, {self.fit_to_crop})"

    def __call__(self, image):
        height = image.shape[0]
        width = image.shape[1]
        image = image.copy()

        image[: int(height * self.zeroing_ratio[0])] = 0
        image[height - int(height * self.zeroing_ratio[1]) :] = 0
        image[:, : int(width * self.zeroing_ratio[2])] = 0
        image[:, width - int(width * self.zeroing_ratio[3]) :] = 0

        if self.fit_to_crop and self.shape != image.shape:
            self.shape = image.shape
            height, width, _ = self.shape
            self.warp.updateMatrixWithPoints(
                [
                    [width * self.zeroing_ratio[2], height * self.zeroing_ratio[0]],
                    [
                        width * (1 - self.zeroing_ratio[3]),
                        height * (1 - self.zeroing_ratio[1]),
                    ],
                    [
                        width * (1 - self.zeroing_ratio[3]),
                        height * self.zeroing_ratio[0],
                    ],
                ],
                [
                    [width * 0, height * 0],
                    [width * 1, height * 1],
                    [width * 1, height * 0],
                ],
            )

        image = self.warp(image)
        return image


if __name__ == "__main__":
    test = np.random.rand(1024, 1024, 3).astype(np.float32)
    augmentor = ZeroingEdge()

    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
