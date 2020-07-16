import numpy as np

from ltr.data import transforms
import ltr.data.processing_utils as prutils
from ltr.data.anchor import AnchorTarget
from pytracking.libs import TensorDict


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self,
                 transform=transforms.ToArray(),
                 train_transform=None,
                 test_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {
            'train': transform if train_transform is None else train_transform,
            'test': transform if test_transform is None else test_transform,
            'joint': joint_transform
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SiamFCProcessing(BaseProcessing):
    def __init__(self,
                 search_area_factor,
                 output_sz,
                 center_jitter_factor,
                 scale_jitter_factor,
                 mode='pair',
                 scale_type='context',
                 border_type='meanpad',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.scale_type = scale_type
        self.border_type = border_type

    def _get_jittered_box(self, box, mode, rng):
        jittered_size = box[2:4] * np.exp(
            rng.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (np.sqrt(jittered_size.prod()) *
                      self.center_jitter_factor[mode])
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (rng.rand(2)
                                                                    - 0.5)

        return np.concatenate(
            (jittered_center - 0.5 * jittered_size, jittered_size), axis=0)

    def __call__(self, data: TensorDict, rng=None):
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [
                self._get_jittered_box(a, s, rng) for a in data[s + '_anno']
            ]

            # Crop image region centered at jittered_anno box
            try:
                crops, boxes = prutils.jittered_center_crop(
                    data[s + '_images'],
                    jittered_anno,
                    data[s + '_anno'],
                    self.search_area_factor[s],
                    self.output_sz[s],
                    scale_type=self.scale_type,
                    border_type=self.border_type)
            except Exception as e:
                print('{}, anno: {}'.format(data['dataset'], data[s + '_anno']))
                raise e

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class SiamProcessing(BaseProcessing):
    def __init__(self,
                 search_area_factor,
                 output_sz,
                 center_jitter_factor,
                 scale_jitter_factor,
                 label_params,
                 mode='pair',
                 scale_type='context',
                 border_type='meanpad',
                 *args,
                 **kwargs):
        self._init_transform(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.scale_type = scale_type
        self.border_type = border_type
        self.label_params = label_params
        self.anchor_target = AnchorTarget(
            label_params['search_size'],
            label_params['output_size'],
            label_params['anchor_stride'],
            label_params['anchor_ratios'],
            label_params['anchor_scales'],
            label_params['num_pos'],
            label_params['num_neg'],
            label_params['num_total'],
            label_params['thr_high'],
            label_params['thr_low'])

    def _init_transform(self,
                        transform=transforms.ToArray(),
                        train_transform=None,
                        test_transform=None,
                        train_mask_transform=None,
                        test_mask_transform=None,
                        joint_transform=None):
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test': transform if test_transform is None else test_transform,
                          'joint': joint_transform}
        super().__init__(
            transform=transform,
            train_transform=train_transform,
            test_transform=test_transform,
            joint_transform=joint_transform)
        self.transform['train_mask'] = self.transform['train'] if train_mask_transform is None \
            else train_mask_transform
        self.transform['test_mask'] = self.transform['test'] if test_mask_transform is None \
            else test_mask_transform

    def _get_jittered_box(self, box, mode, rng):
        jittered_size = box[2:4] * (1 + (2 * rng.rand(2) - 1) * self.scale_jitter_factor[mode])
        max_offset = (np.sqrt(jittered_size.prod()) * self.center_jitter_factor[mode])
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (rng.rand(2) - 0.5)

        return np.concatenate((jittered_center - 0.5 * jittered_size, jittered_size), axis=0)

    def _get_label(self, target_bb, neg):
        return self.anchor_target(target_bb, self.label_params['output_size'], neg)

    def __call__(self, data: TensorDict, rng=None):
        neg = data['neg']

        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s, rng) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            try:
                crops, boxes = prutils.jittered_center_crop(
                    data[s + '_images'],
                    jittered_anno,
                    data[s + '_anno'],
                    self.search_area_factor[s],
                    self.output_sz[s],
                    scale_type=self.scale_type,
                    border_type=self.border_type)
                mask_crops, _ = prutils.jittered_center_crop(
                    data[s + '_masks'],
                    jittered_anno,
                    data[s + '_anno'],
                    self.search_area_factor[s],
                    self.output_sz[s],
                    scale_type=self.scale_type,
                    border_type='zeropad')
            except Exception as e:
                print('{}, anno: {}'.format(data['dataset'], data[s + '_anno']))
                raise e

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes
            data[s + '_masks'] = [self.transform[s + '_mask'](x) for x in mask_crops]

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Get labels
        if self.label_params is not None:
            assert data['test_anno'].shape[0] == 1
            gt_box = data['test_anno'][0]
            gt_box[2:] += gt_box[:2]
            cls, delta, delta_weight, overlap = self._get_label(gt_box, neg)

            mask = data['test_masks'][0]
            if np.sum(mask) > 0:
                mask_weight = cls.max(axis=0, keepdims=True)
            else:
                mask_weight = np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32)
            mask = (mask > 0.5) * 2. - 1.

            data['label_cls'] = cls
            data['label_loc'] = delta
            data['label_loc_weight'] = delta_weight
            data['label_mask'] = mask
            data['label_mask_weight'] = mask_weight
            data.pop('train_anno')
            data.pop('test_anno')
            data.pop('train_masks')
            data.pop('test_masks')

        return data


class ATOMProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """

    def __init__(self,
                 search_area_factor,
                 output_sz,
                 center_jitter_factor,
                 scale_jitter_factor,
                 proposal_params,
                 mode='pair',
                 *args,
                 **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode, rng):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            Variable - jittered box
        """

        jittered_size = box[2:4] * np.exp(
            rng.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (np.sqrt(jittered_size.prod()) *
                      self.center_jitter_factor[mode])
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (rng.rand(2)
                                                                    - 0.5)

        return np.concatenate(
            (jittered_center - 0.5 * jittered_size, jittered_size), axis=0)

    def _generate_proposals(self, box, rng):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            array - Array of shape (num_proposals, 4) containing proposals
            array - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = np.zeros((num_proposals, 4))
        gt_iou = np.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(
                box,
                min_iou=self.proposal_params['min_iou'],
                sigma_factor=self.proposal_params['sigma_factor'],
                rng=rng)

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict, rng=None):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals'-
                'proposal_iou'  -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [
                self._get_jittered_box(a, s, rng) for a in data[s + '_anno']
            ]

            # Crop image region centered at jittered_anno box
            try:
                crops, boxes = prutils.jittered_center_crop(
                    data[s + '_images'], jittered_anno, data[s + '_anno'],
                    self.search_area_factor, self.output_sz)
            except Exception as e:
                print('{}, anno: {}'.format(data['dataset'], data[s + '_anno']))
                raise e
            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

        # Generate proposals
        frame2_proposals, gt_iou = zip(
            * [self._generate_proposals(a, rng) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
