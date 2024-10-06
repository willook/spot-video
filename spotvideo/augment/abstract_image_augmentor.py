import numpy as np


class AbstractImageAugmentor:
    def __call__(self, image):
        raise NotImplementedError

    def getName(self):
        raise NotImplementedError
