from dataclasses import dataclass

import numpy as np

@dataclass
class Result:
    labels: np.ndarray
    predicted_labels: np.ndarray
    similarities: np.ndarray
    threshold: float
    elapsed: float
    acc: float
    f1: float
    