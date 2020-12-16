import numpy as np

from pytracking.features import deep
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.utils import TrackerParams, FeatureParams


def parameters():
    params = TrackerParams()

    # These are usually set from outside
    params.debug = 0  # Debug level
    params.visualization = False  # Do visualization

    # Use GPU or not (IoUNet requires this to be True)
    params.use_gpu = True

    # Feature specific parameters
    deep_params = TrackerParams()

    # Patch sampling parameters
    params.exemplar_size = 127
    params.instance_size = 255
    params.base_size = 8
    params.context_amount = 0.5
    params.mask_output_size = 127
    
    # Anchor parameters
    params.anchor_stride = 8
    params.anchor_ratios = [0.33, 0.5, 1, 2, 3]
    params.anchor_scales = [8]

    # Tracking parameters
    params.penalty_k = 0.20
    params.window_influence = 0.41
    params.lr = 0.30
    params.mask_threshold = 0.30
    
    # output polygon result
    params.polygon = True

    # Setup the feature extractor
    deep_fparams = FeatureParams(feature_params=[deep_params])
    deep_feat = deep.SMaskResNet50_sharp(fparams=deep_fparams)
    params.features = MultiResolutionExtractor([deep_feat])

    return params
