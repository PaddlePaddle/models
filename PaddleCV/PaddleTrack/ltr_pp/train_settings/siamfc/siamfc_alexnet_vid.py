import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

import ltr_pp.actors as actors
import ltr_pp.data.transforms as dltransforms
from ltr_pp.data import processing, sampler, loader
from ltr_pp.dataset import ImagenetVID, Got10k
from ltr_pp.models.siamese.siam import siamfc_alexnet
from ltr_pp.trainers import LTRTrainer
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance


class DataAug(dltransforms.Transform):
    def __init__(self):
        pass

    def random_blur(self, img):
        k = np.random.choice([3, 5, 7])
        return cv.GaussianBlur(img, (k, k), sigmaX=0, sigmaY=0)

    def brightness(self, img):
        img = Image.fromarray(img.astype('uint8'))
        enh_bri = ImageEnhance.Brightness(img)
        brightness = np.random.choice(np.linspace(0.5, 1.25, 4))
        img_brighted = enh_bri.enhance(brightness)

        return np.array(img_brighted)

    def contrast(self, img):
        img = Image.fromarray(img.astype('uint8'))
        enh_con = ImageEnhance.Contrast(img)
        contrast = np.random.choice(np.linspace(0.5, 1.25, 4))
        image_contrasted = enh_con.enhance(contrast)

        return np.array(image_contrasted)

    def no_aug(self, img):
        return img

    def flip(self, img):
        return cv.flip(img, 1)

    def transform(self, img, *args):
        func = np.random.choice([self.contrast, self.random_blur, self.brightness, self.flip])
        return func(img)


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'SiamFC with Alexnet backbone and trained with vid'
    settings.print_interval = 100  # How often to print loss and other info
    settings.batch_size = 8  # Batch size
    settings.num_workers = 8  # Number of workers for image loading
    settings.normalize_mean = [0., 0., 0.]  # Normalize mean
    settings.normalize_std = [1 / 255., 1 / 255., 1 / 255.]  # Normalize std
    settings.search_area_factor = {'train': 1.0, 'test': 2.0078740157480315}  # roughly the same as SiamFC
    settings.output_sz = {'train': 127, 'test': 255}
    settings.scale_type = 'context'
    settings.border_type = 'meanpad'

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 0}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.}

    # Train datasets
    vid_train = ImagenetVID()

    # Validation datasets
    got10k_val = vid_train#Got10k(split='val')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.25)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_exemplar = dltransforms.Compose([dltransforms.ToArray(),
                                               dltransforms.Normalize(mean=settings.normalize_mean,
                                                                      std=settings.normalize_std)])
    transform_instance = dltransforms.Compose([DataAug(),
                                               dltransforms.ToArray(),
                                               dltransforms.Normalize(mean=settings.normalize_mean,
                                                                      std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = processing.SiamFCProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        scale_type=settings.scale_type,
                                                        border_type=settings.border_type,
                                                        mode='sequence',
                                                        train_transform=transform_exemplar,
                                                        test_transform=transform_instance,
                                                        joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.SiamFCProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      scale_type=settings.scale_type,
                                                      border_type=settings.border_type,
                                                      mode='sequence',
                                                      transform=transform_exemplar,
                                                      joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([vid_train], [1, ],
                                        samples_per_epoch=6650 * settings.batch_size, max_gap=100,
                                        processing=data_processing_train)

    # The loader for training
    train_loader = loader.LTRLoader('train', dataset_train,
                                    training=True,
                                    batch_size=settings.batch_size,
                                    num_workers=settings.num_workers,
                                    stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ATOMSampler([got10k_val], [1, ],
                                      samples_per_epoch=1000 * settings.batch_size, max_gap=100,
                                      processing=data_processing_val)

    # The loader for validation
    val_loader = loader.LTRLoader('val', dataset_val,
                                  training=False,
                                  batch_size=settings.batch_size,
                                  num_workers=settings.num_workers,
                                  epoch_interval=5,
                                  stack_dim=1)

    # creat network, set objective, creat optimizer, learning rate scheduler, trainer
    with dygraph.guard():
        # Create network
        net = siamfc_alexnet()

        # Create actor, which wraps network and objective
        actor = actors.SiamFCActor(net=net, objective=None,
                                   batch_size=settings.batch_size,
                                   shape=(17, 17),
                                   radius=16,
                                   stride=8)

        # Set to training mode
        actor.train()

        # define optimizer and learning rate
        lr_scheduler = fluid.layers.exponential_decay(learning_rate=0.01,
                                                      decay_steps=6650,
                                                      decay_rate=0.8685,
                                                      staircase=True)
        regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0005)
        optimizer = fluid.optimizer.Momentum(
            momentum=0.9, regularization=regularizer,
            parameter_list=net.parameters(),
            learning_rate=lr_scheduler)

        trainer = LTRTrainer(actor, [train_loader], optimizer, settings, lr_scheduler)
        trainer.train(50, load_latest=False, fail_safe=False)
