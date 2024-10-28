import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # fast blur
    blur = cv2.resize(
        gray, fx=0.37, fy=0.37, dsize=(0, 0), interpolation=cv2.INTER_LINEAR
    )
    return blur
