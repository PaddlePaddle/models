"""
    @author fangyi.zhang@vipl.ict.ac.cn
"""
import numpy as np

def determine_thresholds(confidence, resolution=100):
    """choose threshold according to confidence

    Args:
        confidence: list or numpy array or numpy array
        reolution: number of threshold to choose

    Restures:
        threshold: numpy array
    """
    if isinstance(confidence, list):
        confidence = np.array(confidence)
    confidence = confidence.flatten()
    confidence = confidence[~np.isnan(confidence)]
    confidence.sort()

    assert len(confidence) > resolution and resolution > 2

    thresholds = np.ones((resolution))
    thresholds[0] = - np.inf
    thresholds[-1] = np.inf
    delta = np.floor(len(confidence) / (resolution - 2))
    idxs = np.linspace(delta, len(confidence)-delta, resolution-2, dtype=np.int32)
    thresholds[1:-1] =  confidence[idxs]
    return thresholds
