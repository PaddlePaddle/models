import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

import ltr.actors as actors
import ltr.data.transforms as dltransforms
from ltr.data import processing, sampler, loader
from ltr.dataset import ImagenetVID, ImagenetDET, MSCOCOSeq, YoutubeVOS
from ltr.models.siam.siam import SiamMask_ResNet50_sharp
from ltr.models.loss import select_softmax_with_cross_entropy_loss, weight_l1_loss, select_mask_logistic_loss 
from ltr.trainers import LTRTrainer
from ltr.trainers.learning_rate_scheduler import LinearLrWarmup
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.base_model = ''
    settings.description = 'SiamMask_sharp with ResNet-50 backbone.'
    settings.print_interval = 100  # How often to print loss and other info
    settings.batch_size = 64  # Batch size
    settings.samples_per_epoch = 600000 # Number of training pairs per epoch
    settings.num_workers = 8  # Number of workers for image loading
    settings.search_area_factor = {'train': 1.0, 'test': 143./127.}
    settings.output_sz = {'train': 127, 'test': 143}
    settings.scale_type = 'context'
    settings.border_type = 'meanpad'

    # Settings for the image sample and label generation
    settings.center_jitter_factor = {'train': 0.2, 'test': 0.4}
    settings.scale_jitter_factor = {'train': 0.05, 'test': 0.18}
    settings.label_params = {
        'search_size': 143,
        'output_size': 3,
        'anchor_stride': 8,
        'anchor_ratios': [0.33, 0.5, 1, 2, 3],
        'anchor_scales': [8],
        'num_pos': 16,
        'num_neg': 16,
        'num_total': 64,
        'thr_high': 0.6,
        'thr_low': 0.3
    }
    settings.loss_weights = {'cls': 0., 'loc': 0., 'mask':1}
    settings.neg = 0

    # Train datasets
    vos_train = YoutubeVOS()
    coco_train = MSCOCOSeq()

    # Validation datasets
    vos_val = vos_train

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.25)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_exemplar = dltransforms.Transpose()
    transform_instance = dltransforms.Compose(
        [
            dltransforms.Color(probability=1.0),
            dltransforms.Blur(probability=0.18),
            dltransforms.Transpose()
        ])
    transform_instance_mask = dltransforms.Transpose()

    # Data processing to do on the training pairs
    data_processing_train = processing.SiamProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        scale_type=settings.scale_type,
        border_type=settings.border_type,
        mode='sequence',
        label_params=settings.label_params,
        train_transform=transform_exemplar,
        test_transform=transform_instance,
        test_mask_transform=transform_instance_mask,
        joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.SiamProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        scale_type=settings.scale_type,
        border_type=settings.border_type,
        mode='sequence',
        label_params=settings.label_params,
        transform=transform_exemplar,
        joint_transform=transform_joint)

    nums_per_epoch = settings.samples_per_epoch // settings.batch_size
    # The sampler for training
    dataset_train = sampler.MaskSampler(
        [coco_train, vos_train],
        [1 ,1],
        samples_per_epoch=nums_per_epoch * settings.batch_size,
        max_gap=100,
        processing=data_processing_train,
        neg=settings.neg)

    # The loader for training
    train_loader = loader.LTRLoader(
        'train',
        dataset_train,
        training=True,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        stack_dim=0)

    # The sampler for validation
    dataset_val = sampler.MaskSampler(
        [vos_val],
        [1, ],
        samples_per_epoch=100 * settings.batch_size,
        max_gap=100,
        processing=data_processing_val)

    # The loader for validation
    val_loader = loader.LTRLoader(
        'val',
        dataset_val,
        training=False,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        stack_dim=0)

    # creat network, set objective, creat optimizer, learning rate scheduler, trainer
    with dygraph.guard():
        # Create network

        def scale_loss(loss):
            total_loss = 0
            for k in settings.loss_weights:
                total_loss += loss[k] * settings.loss_weights[k]
            return total_loss
        
        net = SiamMask_ResNet50_sharp(scale_loss=scale_loss)

        # Load parameters from the best_base_model
        if settings.base_model == '':
            raise Exception(
                'The base_model path is not setup. Check settings.base_model in "ltr/train_settings/siammask/siammask_res50_sharp.py".'
            )
        para_dict, _ = fluid.load_dygraph(settings.base_model)
        model_dict = net.state_dict()

        for key in model_dict.keys():
            if key in para_dict.keys():
                model_dict[key] = para_dict[key]

        net.set_dict(model_dict)

        # Define objective
        objective = {
            'cls': select_softmax_with_cross_entropy_loss,
            'loc': weight_l1_loss,
            'mask': select_mask_logistic_loss
        }

        # Create actor, which wraps network and objective
        actor = actors.SiamActor(net=net, objective=objective)

        # Set to training mode
        actor.train()

        # Define optimizer and learning rate
        decayed_lr = fluid.layers.exponential_decay(
            learning_rate=0.0005,
            decay_steps=nums_per_epoch,
            decay_rate=0.9,
            staircase=True)
        lr_scheduler = LinearLrWarmup(
            learning_rate=decayed_lr,
            warmup_steps=5*nums_per_epoch,
            start_lr=0.0001,
            end_lr=0.0005)

        optimizer = fluid.optimizer.Adam(
            parameter_list=net.mask_head.parameters()
                           + net.refine_head.parameters(),
            learning_rate=lr_scheduler)

        trainer = LTRTrainer(actor, [train_loader, val_loader], optimizer, settings, lr_scheduler)
        trainer.train(20, load_latest=False, fail_safe=False)
