from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class Tile(AbstractImageAugmentor):
    def __init__(self, tile_num=2):
        self.tile_num = tile_num

    def __call__(self, image):

        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Resize the image to a smaller size
        small_size = original_width // self.tile_num, original_height // self.tile_num
        small_image = cv2.resize(image, small_size)


        # Create a new image by tiling the small image
        tiled_image = np.tile(small_image, (self.tile_num, self.tile_num, 1))

        # Resize the tiled image to match the original dimensions
        tiled_image = cv2.resize(tiled_image, (original_width, original_height))

        return tiled_image


if __name__ == '__main__':
    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    test = np.sin(1.5 * np.pi * (xx + yy))

    test = np.stack([test, test, test], axis=-1).astype(np.float32)
    test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)

    augmentor = Tile()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
