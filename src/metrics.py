"""Metrics written by hand (Tasks 3 and 5)."""
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """2x2 array. Rows = true class (0, 1), columns = predicted class (0, 1).

    [[TN, FP],
     [FN, TP]]
    """
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])
