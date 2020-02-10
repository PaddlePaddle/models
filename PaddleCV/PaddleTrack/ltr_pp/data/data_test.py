import unittest

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from ltr_pp.data import processing, sampler, loader
from ltr_pp.dataset import ImagenetVID
import ltr_pp.data.transforms as dltransforms


class TestSampler(unittest.TestCase):
    def test_sampler(self):
        import ltr_pp.admin.settings as ws_settings
        settings = ws_settings.Settings()

        settings.batch_size = 64
        settings.search_area_factor = 5.0
        settings.output_sz = 16 * 18

        settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
        settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)

        # Settings for the image sample and proposal generation
        settings.center_jitter_factor = {'train': 0, 'test': 4.5}
        settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
        settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

        # The joint augmentation transform, that is applied to the pairs jointly
        transform_joint = dltransforms.ToGrayscale(probability=0.05)

        # The augmentation transform applied to the training set (individually to each image in the pair)
        transform_train = dltransforms.Compose([dltransforms.ToArrayAndJitter(0.2),
                                                dltransforms.Normalize(mean=settings.normalize_mean,
                                                                       std=settings.normalize_std)])

        # The augmentation transform applied to the validation set (individually to each image in the pair)
        transform_val = dltransforms.Compose([dltransforms.ToArray(),
                                              dltransforms.Normalize(mean=settings.normalize_mean,
                                                                     std=settings.normalize_std)])

        vid_train = ImagenetVID()
        data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                          output_sz=settings.output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          proposal_params=settings.proposal_params,
                                                          transform=transform_train,
                                                          joint_transform=transform_joint)

        dataset_train = sampler.ATOMSampler([vid_train], [1, ],
                                            samples_per_epoch=1000 * settings.batch_size, max_gap=50,
                                            processing=data_processing_train)

        dataset_train.reset_state()
        one_data = dataset_train.__iter__()

    def test_loader(self):
        import ltr_pp.admin.settings as ws_settings
        import dataflow as df

        settings = ws_settings.Settings()

        settings.batch_size = 64
        settings.search_area_factor = 5.0
        settings.output_sz = 16 * 18

        settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
        settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)

        # Settings for the image sample and proposal generation
        settings.center_jitter_factor = {'train': 0, 'test': 4.5}
        settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
        settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

        # The joint augmentation transform, that is applied to the pairs jointly
        transform_joint = dltransforms.ToGrayscale(probability=0.05)

        # The augmentation transform applied to the training set (individually to each image in the pair)
        transform_train = dltransforms.Compose([dltransforms.ToArrayAndJitter(0.2),
                                                dltransforms.Normalize(mean=settings.normalize_mean,
                                                                       std=settings.normalize_std)])

        # The augmentation transform applied to the validation set (individually to each image in the pair)
        transform_val = dltransforms.Compose([dltransforms.ToArray(),
                                              dltransforms.Normalize(mean=settings.normalize_mean,
                                                                     std=settings.normalize_std)])

        vid_train = ImagenetVID()
        data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                          output_sz=settings.output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          proposal_params=settings.proposal_params,
                                                          transform=transform_train,
                                                          joint_transform=transform_joint)

        dataset_train = sampler.ATOMSampler([vid_train], [1, ],
                                            samples_per_epoch=1000 * settings.batch_size, max_gap=50,
                                            processing=data_processing_train)

        train_loader = loader.LTRLoader('train', dataset_train,
                                        batch_size=64, num_workers=10)

        test_loader = df.TestDataSpeed(train_loader, 1000)
        test_loader.start()

if __name__ == '__main__':
    unittest.main()
