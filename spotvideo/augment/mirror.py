from abstract_image_augmentor import AbstractImageAugmentor
import numpy as np
import cv2

from matplotlib import pyplot as plt

class Mirror(AbstractImageAugmentor):
    # y = ax + b
    def __init__(self, a:float=None, b:float=None):
        if a is None:
            a = (np.random.rand() - 0.5) * 10
        if b is None:
            b = np.random.rand()
            eps = 0.01
            if a > 0:
                # -a * 1 + 0 < b < -a * 0 + 1
                b = b * (-a * 0 + 1 - -a * 1 + 0 - eps) + -a * 1 + 0 + eps
            else:
                # -a * 0 + 0 < b < -a * 1 + 1
                b = b * (-a * 1 + 1 - -a * 0 + 0 - eps) + -a * 0 + 0 + eps

        self.fixUpper = True if a * 0.5 + b - 0.5 < 0 else False

        width = 1000
        height = 1000
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        x1, y1 = np.meshgrid(x, y)

        # 주어진 점 P(x1, y1)
    
        # 직선 y = ax + b에 대해 수직선의 방정식
        # y - y1 = -1/a * (x - x1) => ax + y = ax1 + y1
        # 두 방정식을 동시에 풀기
        # y = ax + b
        # y = -1/a * (x - x1) + y1
        
        # 교차점 Q(x_q, y_q)
        x_q = (a * (y1 - b) + x1) / (a**2 + 1)
        y_q = a * x_q + b

        # 대칭점 P'(x', y')
        x_prime = 2 * x_q - x1
        y_prime = 2 * y_q - y1

        self.x = x1
        self.y = y1

        if self.fixUpper:
            mask = a * x1 + b - y1 > 0
            self.x[mask] = x_prime[mask]
            self.y[mask] = y_prime[mask]
        else:
            mask = a * x1 + b - y1 < 0
            self.x[mask] = x_prime[mask]
            self.y[mask] = y_prime[mask]
        
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __call__(self, image):
        height, width = image.shape[:2]
        map_x = cv2.resize(self.x, (width, height)) * (width - 1)
        map_y = cv2.resize(self.y, (width, height)) * (height - 1)
        
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return image

if __name__ == '__main__':
    width = 200
    height = 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    test = np.sin(2 * np.pi * (xx + yy))

    test = np.stack([test, test, test], axis=-1).astype(np.float32)
    test = cv2.normalize(test, None, 0, 1.0, cv2.NORM_MINMAX)

    augmentor = Mirror()
    
    output = augmentor(test)

    plt.subplot(1, 2, 1)
    plt.imshow(test)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
