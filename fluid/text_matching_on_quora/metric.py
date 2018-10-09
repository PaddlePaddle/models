import numpy as np

"""
This Module defines evaluate metrics for classification tasks
"""


def accuracy(y_pred, label):
    """
    define correct: the top 1 class in y_pred is the same as y_true
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = np.argmax(y_pred, axis=1)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]

def accuracy_with_threshold(y_pred, label, threshold=0.5):
    """
    define correct: the y_true class's prob in y_pred is bigger than threshold
    when threshold is 0.5, This fuction is equal to accuracy
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = (y_pred[:, 1] > threshold).astype(int)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]
