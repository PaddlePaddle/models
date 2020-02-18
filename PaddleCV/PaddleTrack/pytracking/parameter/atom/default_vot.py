import numpy as np

from pytracking.features.deep import ResNet50
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
    params.max_image_sample_size = (14 * 16) ** 2  # Maximum image sample size
    params.min_image_sample_size = (14 * 16) ** 2  # Minimum image sample size
    params.search_area_scale = 4  # Scale relative to target size
    params.feature_size_odd = False  # Good to use False for even-sized kernels and vice versa

    # Optimization parameters
    params.CG_iter = 5  # The number of Conjugate Gradient iterations in each update after the first frame
    params.init_CG_iter = 60  # The total number of Conjugate Gradient iterations used in the first frame
    params.init_GN_iter = 6  # The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
    params.post_init_CG_iter = 0  # CG iterations to run after GN
    params.fletcher_reeves = False  # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
    params.standard_alpha = True  # Use the standard formula for computing the step length in Conjugate Gradient
    params.CG_forgetting_rate = None  # Forgetting rate of the last conjugate direction

    # Learning parameters for each feature type
    deep_params.learning_rate = 0.0075  # Learning rate
    deep_params.output_sigma_factor = 1 / 4  # Standard deviation of Gaussian label relative to target size

    # Training parameters
    params.sample_memory_size = 250  # Memory size
    params.train_skipping = 10  # How often to run training (every n-th frame)

    # Online model parameters
    deep_params.kernel_size = (4, 4)  # Kernel size of filter
    deep_params.compressed_dim = 64  # Dimension output of projection matrix
    deep_params.filter_reg = 1e-1  # Filter regularization factor
    deep_params.projection_reg = 1e-4  # Projection regularization factor

    # Windowing
    params.feature_window = False  # Perform windowing of features
    params.window_output = True  # Perform windowing of output scores

    # Detection parameters
    params.scale_factors = np.array([1],
                                    dtype='float32')  # What scales to use for localization (only one scale if IoUNet is used)
    params.score_upsample_factor = 1  # How much Fourier upsampling to use

    # Init data augmentation parameters
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           'dropout': (7, 0.2)}

    params.augmentation_expansion_factor = 2  # How much to expand sample when doing augmentation
    params.random_shift_factor = 1 / 3  # How much random shift to do on each augmented sample
    deep_params.use_augmentation = True  # Whether to use augmentation for this feature

    # Factorized convolution parameters
    # params.use_projection_matrix = True       # Use projection matrix, i.e. use the factorized convolution formulation
    params.update_projection_matrix = True  # Whether the projection matrix should be optimized or not
    params.proj_init_method = 'randn'  # Method for initializing the projection matrix
    params.filter_init_method = 'randn'  # Method for initializing the spatial filter
    params.projection_activation = 'none'  # Activation function after projection ('none', 'relu', 'elu' or 'mlu')
    params.response_activation = (
    'mlu', 0.05)  # Activation function on the output scores ('none', 'relu', 'elu' or 'mlu')

    # Advanced localization parameters
    params.advanced_localization = True  # Use this or not
    params.target_not_found_threshold = -1  # Absolute score threshold to detect target missing
    params.distractor_threshold = 100  # Relative threshold to find distractors
    params.hard_negative_threshold = 0.3  # Relative threshold to find hard negative samples
    params.target_neighborhood_scale = 2.2  # Target neighborhood to remove
    params.dispalcement_scale = 0.7  # Dispacement to consider for distractors
    params.hard_negative_learning_rate = 0.02  # Learning rate if hard negative detected
    params.hard_negative_CG_iter = 5  # Number of optimization iterations to use if hard negative detected
    params.update_scale_when_uncertain = True  # Update scale or not if distractor is close

    # IoUNet parameters
    params.iounet_augmentation = False  # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3  # Top-k average to estimate final box
    params.num_init_random_boxes = 9  # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1  # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5  # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6  # Limit on the aspect ratio
    params.box_refinement_iter = 5  # Number of iterations for refining the boxes
    params.box_refinement_step_length = 1  # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1  # Multiplicative step length decay (1 means no decay)

    # Setup the feature extractor (which includes the IoUNet)
    deep_fparams = FeatureParams(feature_params=[deep_params])
    deep_feat = ResNet50(net_path='/home/vis/bily/code/baidu/personal-code/libi-13/paddle_ATOMnet-ep0040',
                         output_layers=['block2'],
                         fparams=deep_fparams,
                         normalize_power=2)
    params.features = MultiResolutionExtractor([deep_feat])

    params.vot_anno_conversion_type = 'preserve_area'
    return params
