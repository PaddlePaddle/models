import math
import os
import time

import numpy as np
from paddle import fluid
from paddle.fluid import layers

from pytracking.features import augmentation
from pytracking.libs import dcf, operation, fourier
from pytracking.libs.optimization import ConjugateGradient, GaussNewtonCG, GradientDescentL2
from pytracking.libs.paddle_utils import mod, n2p, \
    leaky_relu, dropout2d
from pytracking.libs.tensorlist import TensorList
from pytracking.tracker.atom.optim import FactorizedConvProblem, ConvProblem
from pytracking.tracker.base.basetracker import BaseTracker


class ATOM(BaseTracker):
    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True

    def initialize(self, image, state, *args, **kwargs):
        # Initialize some stuff
        self.frame_num = 1
        # TODO: for now, we don't support explictly setting up device
        # if not hasattr(self.params, 'device'):
        #     self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features
        self.initialize_features()

        # Check if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        self.time = 0
        tic = time.time()

        # Get position and size
        self.pos = np.array(
            [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2],
            'float32')
        self.target_sz = np.array([state[3], state[2]], 'float32')

        # Set search area
        self.target_scale = 1.0
        search_area = np.prod(self.target_sz * self.params.search_area_scale)
        if search_area > self.params.max_image_sample_size:
            self.target_scale = math.sqrt(search_area /
                                          self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale = math.sqrt(search_area /
                                          self.params.min_image_sample_size)

        # Check if IoUNet is used
        self.use_iou_net = getattr(self.params, 'use_iou_net', True)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        if getattr(self.params, 'search_area_shape', 'square') == 'square':
            self.img_sample_sz = np.ones((2, ), 'float32') * np.round(
                np.sqrt(
                    np.prod(self.base_target_sz *
                            self.params.search_area_scale)))
        elif self.params.search_area_shape == 'initrect':
            self.img_sample_sz = np.round(self.base_target_sz *
                                          self.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.params.feature_size_odd:
            self.img_sample_sz += feat_max_stride - mod(self.img_sample_sz,
                                                        (2 * feat_max_stride))
        else:
            self.img_sample_sz += feat_max_stride - mod(
                (self.img_sample_sz + feat_max_stride), (2 * feat_max_stride))

        # Set sizes
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')

        self.iou_img_sample_sz = self.img_sample_sz

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute(
            'learning_rate')
        if self.params.CG_forgetting_rate is None or max(
                self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (
                1 - max(self.params.precond_learning_rate)
            )**self.params.CG_forgetting_rate

        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(
                    self.output_sz.astype('long'),
                    self.output_sz.astype('long') *
                    self.params.effective_search_area /
                    self.params.search_area_scale,
                    centered=False)
            else:
                self.output_window = dcf.hann2d(
                    self.output_sz.astype('long'), centered=False)

        # Initialize some learning things
        self.init_learning()

        # Convert image
        im = image.astype('float32')
        self.im = im  # For debugging only

        # Setup scale bounds
        self.image_sz = np.array([im.shape[0], im.shape[1]], 'float32')
        self.min_scale_factor = np.max(10 / self.base_target_sz)
        self.max_scale_factor = np.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)

        # Initialize iounet
        if self.use_iou_net:
            self.init_iou_net()

        # Initialize projection matrix
        self.init_projection_matrix(x)

        # Transform to get the training sample
        train_x = self.preprocess_sample(x)

        # Generate label function
        init_y = self.init_label_function(train_x)

        # Init memory
        self.init_memory(train_x)

        # Init optimizer and do initial optimization
        self.init_optimization(train_x, init_y)

        self.pos_iounet = self.pos.copy()

        self.time += time.time() - tic

    def track(self, image):

        self.frame_num += 1

        # Convert image
        # im = numpy_to_paddle(image)
        im = image.astype('float32')
        self.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors

        test_x = self.extract_processed_sample(im, self.pos, sample_scales,
                                               self.img_sample_sz)

        # Compute scores
        scores_raw = self.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)

        # Update position and scale
        if flag != 'not_found':
            if self.use_iou_net:
                update_scale_flag = getattr(self.params,
                                            'update_scale_when_uncertain',
                                            True) or flag != 'uncertain'
                if getattr(self.params, 'use_classifier', True):
                    self.update_state(sample_pos + translation_vec)
                self.refine_target_box(sample_pos, sample_scales[scale_ind],
                                       scale_ind, update_scale_flag)
            elif getattr(self.params, 'use_classifier', True):
                self.update_state(sample_pos + translation_vec,
                                  sample_scales[scale_ind])

        # ------- UPDATE ------- #

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind + 1] for x in test_x])

            # Create label for sample
            train_y = self.get_label_function(sample_pos,
                                              sample_scales[scale_ind])

            # Update memory
            self.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(self.params.hard_negative_CG_iter)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            self.filter_optimizer.run(self.params.CG_iter)
        self.filter = self.filter_optimizer.x

        # Set the pos of the tracker to iounet pos
        if self.use_iou_net and flag != 'not_found':
            self.pos = self.pos_iounet.copy()

        # Return new state
        yx = self.pos - (self.target_sz - 1) / 2
        new_state = np.array(
            [yx[1], yx[0], self.target_sz[1], self.target_sz[0]], 'float32')

        return new_state.tolist()

    def update_memory(self,
                      sample_x: TensorList,
                      sample_y: TensorList,
                      learning_rate=None):
        replace_ind = self.update_sample_weights(
            self.sample_weights, self.previous_replace_ind,
            self.num_stored_samples, self.num_init_samples, self.fparams,
            learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x,
                                      replace_ind):
            train_samp[ind] = x[0]
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind] = y[0]
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind] = layers.cast(y >= self.params.hinge_threshold,
                                     'float32')[0]
        self.num_stored_samples += 1

    def update_sample_weights(self,
                              sample_weights,
                              previous_replace_ind,
                              num_stored_samples,
                              num_init_samples,
                              fparams,
                              learning_rate=None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(
                sample_weights, previous_replace_ind, num_stored_samples,
                num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight',
                                       None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                r_ind = np.argmin(sw[s_ind:], 0)
                r_ind = int(r_ind + s_ind)

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum(
            ) < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.fparams.attribute('translation_weight', 1.0)
        scores_raw = weight * scores_raw
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) *
                                                   scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (
                1 - np.array([ksz[0] % 2, ksz[1] % 2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz)

        if self.output_window is not None and not getattr(
                self.params, 'perform_hn_without_windowing', False):
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        scale_ind = np.argmax(max_score, axis=0)[0]
        max_disp = max_disp.astype('float32')

        # Convert to displacements in the base scale
        output_sz = self.output_sz.copy()
        disp = mod((max_disp + output_sz / 2), output_sz) - output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = np.reshape(
            disp[scale_ind].astype('float32'), [-1]) * (
                self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = np.concatenate(
                [scores[..., sz[0] // 2:, :], scores[..., :sz[0] // 2, :]], -2)
            scores = np.concatenate(
                [scores[..., sz[1] // 2:], scores[..., :sz[1] // 2]], -1)

        return translation_vec, scale_ind, scores, None

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = np.clip(new_scale, self.min_scale_factor,
                                        self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = np.maximum(
            np.minimum(new_pos,
                       self.image_sz.astype('float32') - inside_offset),
            inside_offset)

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (self.img_support_sz *
                                                        sample_scale)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * np.array(
                [(ksz[0] + 1) % 2, (ksz[1] + 1) % 2], 'float32')
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def extract_sample(self,
                       im: np.ndarray,
                       pos: np.ndarray,
                       scales,
                       sz: np.ndarray,
                       debug_save_name):
        return self.params.features.extract(im, pos, scales, sz,
                                            debug_save_name)

    def extract_processed_sample(self,
                                 im: np.ndarray,
                                 pos: np.ndarray,
                                 scales,
                                 sz: np.ndarray,
                                 debug_save_name=None) -> (TensorList,
                                                           TensorList):
        x = self.extract_sample(im, pos, scales, sz, debug_save_name)
        return self.preprocess_sample(self.project_sample(x))

    def apply_filter(self, sample_x: TensorList):
        with fluid.dygraph.guard():
            sample_x = sample_x.apply(n2p)
            filter = self.filter.apply(n2p)
            return operation.conv2d(sample_x, filter, mode='same').numpy()

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(
            self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = self.fparams.attribute('compressed_dim', None)

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                raise NotImplementedError
            elif proj_init_method == 'randn':
                with fluid.dygraph.guard():
                    self.projection_matrix = TensorList([
                        None if cdim is None else layers.gaussian_random(
                            (cdim, ex.shape[1], 1, 1), 0.0,
                            1 / math.sqrt(ex.shape[1])).numpy()
                        for ex, cdim in zip(x, self.compressed_dim)
                    ])
            elif proj_init_method == 'np_randn':
                rng = np.random.RandomState(0)
                self.projection_matrix = TensorList([
                    None if cdim is None else rng.normal(
                        size=(cdim, ex.shape[1], 1, 1),
                        loc=0.0,
                        scale=1 / math.sqrt(ex.shape[1])).astype('float32')
                    for ex, cdim in zip(x, self.compressed_dim)
                ])
            elif proj_init_method == 'ones':
                self.projection_matrix = TensorList([
                    None if cdim is None else
                    np.ones((cdim, ex.shape[1], 1, 1),
                            'float32') / math.sqrt(ex.shape[1])
                    for ex, cdim in zip(x, self.compressed_dim)
                ])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None] * len(x))

    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        if getattr(self.params, '_feature_window', False):
            x = x * self.feature_window
        return x

    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList([
            np.zeros(
                [self.params.sample_memory_size, 1, x.shape[2], x.shape[3]],
                'float32') for x in train_x
        ])

        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        self.sigma = output_sigma_factor * np.ones((2, ), 'float32') * (
            self.feature_sz / self.img_support_sz *
            self.base_target_sz).apply(np.prod).apply(np.sqrt)

        # Center pos in normalized coords
        target_center_norm = (self.pos - np.round(self.pos)) / (
            self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz,
                                      self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * np.array(
                [(ksz[0] + 1) % 2, (ksz[1] + 1) % 2], 'float32')
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + np.array(
                    T.shift, 'float32') / self.img_support_sz * sz
                y[i] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0]] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList(
            [np.ones(x.shape[0], 'float32') / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([
            np.zeros(self.params.sample_memory_size, 'float32') for x in train_x
        ])
        for sw, init_sw, num in zip(self.sample_weights,
                                    self.init_sample_weights,
                                    self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [[np.zeros([cdim, x.shape[2], x.shape[3]], 'float32')] *
             self.params.sample_memory_size
             for x, cdim in zip(train_x, self.compressed_dim)])

    def init_learning(self):
        # Get window function
        self.feature_window = TensorList(
            [dcf.hann2d(sz) for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = self.fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation',
                                        'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = layers.relu
        elif projection_activation == 'elu':
            self.projection_activation = layers.elu
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: layers.elu(leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation',
                                      'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = layers.relu
        elif response_activation == 'elu':
            self.response_activation = layers.elu
        elif response_activation == 'mlu':
            self.response_activation = lambda x: layers.elu(leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

    def generate_init_samples(self, im: np.ndarray) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params,
                                       'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.copy()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz *
                                aug_expansion_factor).astype('long')
            aug_expansion_sz += (
                aug_expansion_sz - self.img_sample_sz.astype('long')) % 2
            aug_expansion_sz = aug_expansion_sz.astype('float32')
            aug_output_sz = self.img_sample_sz.astype('long').tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((np.random.uniform(size=[2]) - 0.5) * self.img_sample_sz * random_shift_factor).astype('long').tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend([
                augmentation.Translation(shift, aug_output_sz)
                for shift in self.params.augmentation['shift']
            ])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (np.array(shift, 'float32') * self.img_sample_sz / 2).astype('long').tolist()
            self.transforms.extend([
                augmentation.Translation(get_absolute(shift), aug_output_sz)
                for shift in self.params.augmentation['relativeshift']
            ])
        if 'fliplr' in self.params.augmentation and self.params.augmentation[
                'fliplr']:
            self.transforms.append(
                augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([
                augmentation.Blur(sigma, aug_output_sz, get_rand_shift())
                for sigma in self.params.augmentation['blur']
            ])
        if 'scale' in self.params.augmentation:
            self.transforms.extend([
                augmentation.Scale(scale_factor, aug_output_sz,
                                   get_rand_shift())
                for scale_factor in self.params.augmentation['scale']
            ])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([
                augmentation.Rotate(angle, aug_output_sz, get_rand_shift())
                for angle in self.params.augmentation['rotate']
            ])

        # Generate initial samples
        init_samples = self.params.features.extract_transformed(
            im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            with fluid.dygraph.guard():
                for i, use_aug in enumerate(
                        self.fparams.attribute('use_augmentation')):
                    if use_aug:
                        init_samples[i] = np.concatenate([
                            init_samples[i], dropout2d(
                                layers.expand(
                                    n2p(init_samples[i][0:1]), (num, 1, 1, 1)),
                                prob,
                                is_train=True).numpy()
                        ])

        return init_samples

    def init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')
        self.filter = TensorList([
            np.zeros([1, cdim, sz[0], sz[1]], 'float32')
            for x, cdim, sz in zip(train_x, self.compressed_dim,
                                   self.kernel_size)
        ])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'ones':
            for idx, f in enumerate(self.filter):
                self.filter[idx] = np.ones(f.shape,
                                           'float32') / np.prod(f.shape)
        elif filter_init_method == 'np_randn':
            rng = np.random.RandomState(0)
            for idx, f in enumerate(self.filter):
                self.filter[idx] = rng.normal(
                    size=f.shape, loc=0,
                    scale=1 / np.prod(f.shape)).astype('float32')
        elif filter_init_method == 'randn':
            for idx, f in enumerate(self.filter):
                with fluid.dygraph.guard():
                    self.filter[idx] = layers.gaussian_random(
                        f.shape, std=1 / np.prod(f.shape)).numpy()
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(
            self.params, 'update_projection_matrix',
            True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')

        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            self.joint_problem = FactorizedConvProblem(
                self.init_training_samples, init_y, self.filter_reg,
                self.fparams.attribute('projection_reg'), self.params,
                self.init_sample_weights, self.projection_activation,
                self.response_activation)

            # Variable containing both filter and projection matrix
            joint_var = self.filter.concat(self.projection_matrix)

            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence',
                                          False)
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(
                    self.joint_problem,
                    joint_var,
                    plotting=(self.params.debug >= 3),
                    analyze=True,
                    fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(
                    self.joint_problem,
                    joint_var,
                    self.params.optimizer_step_length,
                    self.params.optimizer_momentum,
                    plotting=(self.params.debug >= 3),
                    debug=analyze_convergence,
                    fig_num=(12, 13))

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter //
                                         self.params.init_GN_iter,
                                         self.params.init_GN_iter)

            # Get back filter and optimizer
            len_x = len(self.joint_optimizer.x)
            self.filter = self.joint_optimizer.x[:len_x // 2]  # w2 in paper
            self.projection_matrix = self.joint_optimizer.x[len_x //
                                                            2:]  # w1 in paper

            if analyze_convergence:
                opt_name = 'CG' if getattr(self.params, 'CG_optimizer',
                                           True) else 'GD'
                for val_name, values in zip(['loss', 'gradient'], [
                        self.joint_optimizer.losses,
                        self.joint_optimizer.gradient_mags
                ]):
                    val_str = ' '.join(
                        ['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples,
                                                 self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples,
                                         compressed_samples):
            for idx in range(init_samp.shape[0]):
                train_samp[idx] = init_samp[idx]

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y,
                                        self.filter_reg, self.sample_weights,
                                        self.response_activation)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(
                self.conv_problem,
                self.filter,
                fletcher_reeves=self.params.fletcher_reeves,
                direction_forget_factor=self.params.direction_forget_factor,
                debug=(self.params.debug >= 3),
                fig_num=(12, 13))
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(
                self.conv_problem,
                self.filter,
                self.params.optimizer_step_length,
                self.params.optimizer_momentum,
                debug=(self.params.debug >= 3),
                fig_num=12)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)
        self.filter = self.filter_optimizer.x

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer

    def project_sample(self, x: TensorList, proj_matrix=None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix
        with fluid.dygraph.guard():
            return operation.conv2d(x.apply(n2p), proj_matrix.apply(n2p)).apply(
                self.projection_activation).numpy()

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates"""
        box_center = (pos - sample_pos) / sample_scale + (self.iou_img_sample_sz
                                                          - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return np.concatenate([np.flip(target_ul, 0), np.flip(box_sz, 0)])

    def get_iou_features(self):
        return self.params.features.get_unique_attribute('iounet_features')

    def get_iou_backbone_features(self):
        return self.params.features.get_unique_attribute(
            'iounet_backbone_features')

    def init_iou_net(self):
        # Setup IoU net
        self.iou_predictor = self.params.features.get_unique_attribute(
            'iou_predictor')

        # Get target boxes for the different augmentations
        self.iou_target_box = self.get_iounet_box(self.pos, self.target_sz,
                                                  self.pos.round(),
                                                  self.target_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(
                        T, (augmentation.Identity, augmentation.Translation,
                            augmentation.FlipHorizontal,
                            augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.iou_target_box + np.array(
                    [T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.iou_target_box.copy())
        target_boxes = np.concatenate(target_boxes.view(1, 4), 0)

        # Get iou features
        iou_backbone_features = self.get_iou_backbone_features()

        # Remove other augmentations such as rotation
        iou_backbone_features = TensorList(
            [x[:target_boxes.shape[0], ...] for x in iou_backbone_features])

        # Extract target feat
        with fluid.dygraph.guard():
            iou_backbone_features = iou_backbone_features.apply(n2p)
            target_boxes = n2p(target_boxes)
            target_feat = self.iou_predictor.get_filter(iou_backbone_features,
                                                        target_boxes)
            self.target_feat = TensorList(
                [layers.reduce_mean(x, 0).numpy() for x in target_feat])

        if getattr(self.params, 'iounet_not_use_reference', False):
            self.target_feat = TensorList([
                np.full_like(tf, tf.norm() / tf.numel())
                for tf in self.target_feat
            ])

    def optimize_boxes(self, iou_features, init_boxes):
        with fluid.dygraph.guard():
            # Optimize iounet boxes
            init_boxes = np.reshape(init_boxes, (1, -1, 4))
            step_length = self.params.box_refinement_step_length

            target_feat = self.target_feat.apply(n2p)
            iou_features = iou_features.apply(n2p)
            output_boxes = n2p(init_boxes)

            for f in iou_features:
                f.stop_gradient = False
            for i_ in range(self.params.box_refinement_iter):
                # forward pass
                bb_init = output_boxes
                bb_init.stop_gradient = False

                outputs = self.iou_predictor.predict_iou(target_feat,
                                                         iou_features, bb_init)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                outputs.backward()

                # Update proposal
                bb_init_np = bb_init.numpy()
                bb_init_gd = bb_init.gradient()
                output_boxes = bb_init_np + step_length * bb_init_gd * np.tile(
                    bb_init_np[:, :, 2:], (1, 1, 2))
                output_boxes = n2p(output_boxes)
                step_length *= self.params.box_refinement_step_decay

            return layers.reshape(output_boxes, (
                -1, 4)).numpy(), layers.reshape(outputs, (-1, )).numpy()

    def refine_target_box(self,
                          sample_pos,
                          sample_scale,
                          scale_ind,
                          update_scale=True):
        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos,
                                       sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features()
        iou_features = TensorList(
            [x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        init_boxes = np.reshape(init_box, (1, 4)).copy()

        rand_fn = lambda a, b: np.random.rand(a, b).astype('float32')

        if self.params.num_init_random_boxes > 0:
            # Get random initial boxes
            square_box_sz = np.sqrt(init_box[2:].prod())
            rand_factor = square_box_sz * np.concatenate([
                self.params.box_jitter_pos * np.ones(2),
                self.params.box_jitter_sz * np.ones(2)
            ])
            minimal_edge_size = init_box[2:].min() / 3
            rand_bb = (rand_fn(self.params.num_init_random_boxes, 4) - 0.5
                       ) * rand_factor
            new_sz = np.clip(init_box[2:] + rand_bb[:, 2:], minimal_edge_size,
                             1e10)
            new_center = (init_box[:2] + init_box[2:] / 2) + rand_bb[:, :2]
            init_boxes = np.concatenate([new_center - new_sz / 2, new_sz], 1)
            init_boxes = np.concatenate(
                [np.reshape(init_box, (1, 4)), init_boxes])

        # Refine boxes by maximizing iou
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes with extreme aspect ratios
        output_boxes[:, 2:] = np.clip(output_boxes[:, 2:], 1, 1e10)
        aspect_ratio = output_boxes[:, 2] / output_boxes[:, 3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * \
                   (aspect_ratio > 1 / self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind, :]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Take average of top k boxes
        k = getattr(self.params, 'iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        inds = np.argsort(-output_iou)[:topk]
        predicted_box = np.mean(output_boxes[inds, :], axis=0)
        predicted_iou = np.mean(
            np.reshape(output_iou, (-1, 1))[inds, :], axis=0)

        # Update position
        new_pos = predicted_box[:2] + predicted_box[2:] / 2 - (
            self.iou_img_sample_sz - 1) / 2
        new_pos = np.flip(new_pos, 0) * sample_scale + sample_pos
        new_target_sz = np.flip(predicted_box[2:], 0) * sample_scale
        new_scale = np.sqrt(
            np.prod(new_target_sz) / np.prod(self.base_target_sz))

        self.pos_iounet = new_pos.copy()

        if getattr(self.params, 'use_iounet_pos_for_learning', True):
            self.pos = new_pos.copy()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""

        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(
                self.params, 'perform_hn_without_windowing', False):
            scores_orig = scores.copy()

            scores_orig = np.concatenate([
                scores_orig[..., (sz[0] + 1) // 2:, :],
                scores_orig[..., :(sz[0] + 1) // 2, :]
            ], -2)
            scores_orig = np.concatenate([
                scores_orig[..., :, (sz[1] + 1) // 2:],
                scores_orig[..., :, :(sz[1] + 1) // 2]
            ], -1)

            scores *= self.output_window

        # Shift scores back
        scores = np.concatenate([
            scores[..., (sz[0] + 1) // 2:, :], scores[..., :(sz[0] + 1) // 2, :]
        ], -2)
        scores = np.concatenate([
            scores[..., :, (sz[1] + 1) // 2:], scores[..., :, :(sz[1] + 1) // 2]
        ], -1)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        scale_ind = np.argmax(max_score1, axis=0)[0]
        max_score1 = max_score1[scale_ind]
        max_disp1 = np.reshape(max_disp1[scale_ind].astype('float32'), (-1))

        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz /
                                           self.output_sz) * self.target_scale

        if max_score1 < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(
                self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale
        tneigh_top = int(max(round(max_disp1[0] - target_neigh_sz[0] / 2), 0))
        tneigh_bottom = int(
            min(round(max_disp1[0] + target_neigh_sz[0] / 2 + 1), sz[0]))
        tneigh_left = int(max(round(max_disp1[1] - target_neigh_sz[1] / 2), 0))
        tneigh_right = int(
            min(round(max_disp1[1] + target_neigh_sz[1] / 2 + 1), sz[1]))
        scores_masked = scores[scale_ind:scale_ind + 1, ...].copy()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:
                      tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = np.reshape(max_disp2.astype('float32'), (-1))
        target_disp2 = max_disp2 - self.output_sz // 2
        translation_vec2 = target_disp2 * (self.img_support_sz /
                                           self.output_sz) * self.target_scale

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = np.sqrt(np.sum(target_disp1**2))
            disp_norm2 = np.sqrt(np.sum(target_disp2**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(
                sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None
