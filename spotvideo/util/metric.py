import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = sum([1 for x, y in zip(y_true, y_pred) if x == y and x == 1])
    fp = sum([1 for x, y in zip(y_true, y_pred) if x != y and y == 1])
    fn = sum([1 for x, y in zip(y_true, y_pred) if x != y and x == 1])
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    return sum([1 for x, y in zip(y_true, y_pred) if x == y]) / len(y_true)
