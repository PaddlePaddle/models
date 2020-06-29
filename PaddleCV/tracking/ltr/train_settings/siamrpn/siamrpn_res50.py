import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

import ltr.actors as actors
import ltr.data.transforms as dltransforms
from ltr.data import processing, sampler, loader
from ltr.dataset import ImagenetVID, ImagenetDET, MSCOCOSeq, YoutubeVOS
from ltr.models.siam.siam import SiamRPN_ResNet50
from ltr.models.loss import select_softmax_with_cross_entropy_loss, weight_l1_loss
from ltr.trainers import LTRTrainer
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'SiamRPN with ResNet-50 backbone.'
    settings.print_interval = 100  # How often to print loss and other info
    settings.batch_size = 32  # Batch size
    settings.num_workers = 4  # Number of workers for image loading
    settings.search_area_factor = {'train': 1.0, 'test': 255./127.}
    settings.output_sz = {'train': 127, 'test': 255}
    settings.scale_type = 'context'
    settings.border_type = 'meanpad'

    # Settings for the image sample and label generation
    settings.center_jitter_factor = {'train': 0.1, 'test': 1.5}
    settings.scale_jitter_factor = {'train': 0.05, 'test': 0.18}
    settings.label_params = {
        'search_size': 255,
        'output_size': 25,
        'anchor_stride': 8,
        'anchor_ratios': [0.33, 0.5, 1, 2, 3],
        'anchor_scales': [8],
        'num_pos': 16,
        'num_neg': 16,
        'num_total': 64,
        'thr_high': 0.6,
        'thr_low': 0.3
    }
    settings.loss_weights = {'cls': 1., 'loc': 1.2}
    settings.neg = 0.2

    # Train datasets
    vos_train = YoutubeVOS()
    vid_train = ImagenetVID()
    coco_train = MSCOCOSeq()
    det_train = ImagenetDET()

    # Validation datasets
    #vid_val = ImagenetVID()
    vid_val = coco_train

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.25)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_exemplar = dltransforms.Transpose()
    transform_instance = dltransforms.Transpose()

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

    # The sampler for training
    dataset_train = sampler.MaskSampler(
        [vid_train, coco_train, det_train, vos_train],
        [2, 1 ,1, 2],
        samples_per_epoch=5000 * settings.batch_size,
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
        [vid_val],
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
        
        net = SiamRPN_ResNet50(scale_loss=scale_loss)

        # Define objective
        objective = {
            'cls': select_softmax_with_cross_entropy_loss,
            'loc': weight_l1_loss,
        }

        # Create actor, which wraps network and objective
        actor = actors.SiamActor(net=net, objective=objective)

        # Set to training mode
        actor.train()

        # Define optimizer and learning rate
        lr_scheduler = fluid.layers.exponential_decay(
            learning_rate=0.005,
            decay_steps=5000,
            decay_rate=0.9659,
            staircase=True)
        optimizer = fluid.optimizer.Adam(
            parameter_list=net.rpn_head.parameters() + net.neck.parameters(),
            learning_rate=lr_scheduler)

        trainer = LTRTrainer(actor, [train_loader, val_loader], optimizer, settings, lr_scheduler)
        trainer.train(50, load_latest=False, fail_safe=False)
