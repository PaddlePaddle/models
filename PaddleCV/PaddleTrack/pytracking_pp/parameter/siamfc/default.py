import numpy as np

from pytracking_pp.features import deep
from pytracking_pp.features.extractor import MultiResolutionExtractor
from pytracking_pp.utils import TrackerParams, FeatureParams


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
    params.max_image_sample_size = 255*255  # Maximum image sample size
    params.min_image_sample_size = 255*255  # Minimum image sample size

    # Detection parameters
    params.scale_factors = 1.0375 ** np.array([-1, 0, 1])  # What scales to use for localization (only one scale if IoUNet is used)
    params.score_upsample_factor = 16  # How much Fourier upsampling to use
    params.scale_penalty = 0.9745
    params.scale_lr = 0.59
    params.window_influence = 0.176
    params.total_stride = 8

    # Setup the feature extractor (which includes the IoUNet)
    deep_fparams = FeatureParams(feature_params=[deep_params])
    deep_feat = deep.SFCAlexnet(
        net_path='/ssd2/bily/code/baidu/personal-code/pytracking/ltr/checkpoints/ltr/fs/siamrpn50/SiamRPN_ep0001.pth.tar',
        output_layers=['conv5'],
        fparams=deep_fparams)
    params.features = MultiResolutionExtractor([deep_feat])

    params.net_path = ""
    params.response_up = 16
    params.response_sz = 17
    params.context = 0.5
    params.instance_sz = 255
    params.exemplar_sz = 127
    params.scale_num = 3
    params.scale_step = 1.0375
    params.scale_lr = 0.59
    params.scale_penalty = 0.9745
    params.window_influence = 0.176
    params.total_stride = 8
    return params
