import random
import numpy as np
import dataflow as df
from pytracking.libs import TensorDict


def no_processing(data, rng=None):
    return data


class ATOMSampler(df.RNGDataFlow):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a train frame, used to obtain the modulation vector, and ii) a set of test frames on which
    the IoU prediction loss is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A 'train frame' is then sampled randomly from the sequence. Next, depending on the
    frame_sample_mode, the required number of test frames are sampled randomly, either  from the range
    [train_frame_id - max_gap, train_frame_id + max_gap] in the 'default' mode, or from [train_frame_id, train_frame_id + max_gap]
    in the 'causal' mode. Only the frames in which the target is visible are sampled, and if enough visible frames are
    not found, the 'max_gap' is incremented.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self,
                 datasets,
                 p_datasets,
                 samples_per_epoch,
                 max_gap,
                 num_test_frames=1,
                 processing=no_processing,
                 frame_sample_mode='default'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train (reference) frame and the test frames.
            num_test_frames - Number of test frames used for calculating the IoU prediction loss.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'default' or 'causal'. If 'causal', then the test frames are sampled in a causal
                                manner.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = 1  # Only a single train frame allowed
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        inds = self.rng.choice(
            range(len(valid_ids)), size=num_ids, replace=True)
        ids = [valid_ids[ii] for ii in inds]
        # return random.choices(valid_ids, k=num_ids)
        return ids

    def __iter__(self):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        # dataset = self.rng.choices(self.datasets, self.p_datasets)[0]
        dataset_idx = self.rng.choice(
            range(len(self.datasets)), p=self.p_datasets, replace=False)
        dataset = self.datasets[dataset_idx]
        is_video_dataset = dataset.is_video_sequence()

        min_visible_frames = 2 * (self.num_test_frames + self.num_train_frames)
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            seq_id = self.rng.randint(0, dataset.get_num_sequences() - 1)
            anno, visible = dataset.get_sequence_info(seq_id)
            num_visible = np.sum(visible.astype('int64'))
            enough_visible_frames = not is_video_dataset or (
                num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                # Sample frame numbers
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(
                        visible, num_ids=self.num_train_frames)
                    test_frame_ids = self._sample_visible_ids(
                        visible,
                        min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                        max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                        num_ids=self.num_test_frames)
                    gap_increase += 5  # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(
                        visible,
                        num_ids=1,
                        min_id=self.num_train_frames - 1,
                        max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(
                        visible,
                        num_ids=self.num_train_frames - 1,
                        min_id=base_frame_id[0] - self.max_gap - gap_increase,
                        max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(
                        visible,
                        min_id=train_frame_ids[0] + 1,
                        max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                        num_ids=self.num_test_frames)
                    gap_increase += 5  # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        # Get frames
        train_frames, train_anno, _ = dataset.get_frames(seq_id,
                                                         train_frame_ids, anno)
        test_frames, test_anno, _ = dataset.get_frames(seq_id, test_frame_ids,
                                                       anno)

        # Prepare data
        data = TensorDict({
            'train_images': train_frames,
            'train_anno': train_anno,
            'test_images': test_frames,
            'test_anno': test_anno,
            'dataset': dataset.get_name()
        })

        # Send for processing
        yield self.processing(data, rng=self.rng)


class MaskSampler(df.RNGDataFlow):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a train frame, used to obtain the modulation vector, and ii) a set of test frames on which
    the IoU prediction loss is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A 'train frame' is then sampled randomly from the sequence. Next, depending on the
    frame_sample_mode, the required number of test frames are sampled randomly, either  from the range
    [train_frame_id - max_gap, train_frame_id + max_gap] in the 'default' mode, or from [train_frame_id, train_frame_id + max_gap]
    in the 'causal' mode. Only the frames in which the target is visible are sampled, and if enough visible frames are
    not found, the 'max_gap' is incremented.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self,
		 datasets,
		 p_datasets,
		 samples_per_epoch,
		 max_gap,
		 num_test_frames=1,
                 processing=no_processing,
                 frame_sample_mode='default',
                 neg=0):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train (reference) frame and the test frames.
            num_test_frames - Number of test frames used for calculating the rpn/mask prediction loss.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'default' or 'causal'. If 'causal', then the test frames are sampled in a causal
                                manner.
            neg - Probability of sampling a negative sample pair.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = 1  # Only a single train frame allowed
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.neg = neg

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        inds = self.rng.choice(range(len(valid_ids)), size=num_ids, replace=True)
        ids = [valid_ids[ii] for ii in inds]
        # return random.choices(valid_ids, k=num_ids)
        return ids

    def has_mask(self, dataset):
        return dataset.get_name() in ['coco', 'youtubevos']

    def _get_positive_pair(self, dataset):
        is_video_dataset = dataset.is_video_sequence()

        min_visible_frames = 2 * (self.num_test_frames + self.num_train_frames)
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            seq_id = self.rng.randint(0, dataset.get_num_sequences() - 1)
            anno, visible = dataset.get_sequence_info(seq_id)
            num_visible = np.sum(visible.astype('int64'))
            enough_visible_frames = not is_video_dataset or (
                num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                # Sample frame numbers
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames)
                    test_frame_ids = self._sample_visible_ids(
                        visible,
                        min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                        max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                        num_ids=self.num_test_frames)
                    gap_increase += 5  # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(
                        visible,
                        num_ids=1,
                        min_id=self.num_train_frames - 1,
                        max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(
                        visible, num_ids=self.num_train_frames - 1,
                        min_id=base_frame_id[0] - self.max_gap - gap_increase,
                        max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(
                        visible, min_id=train_frame_ids[0] + 1,
                        max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                        num_ids=self.num_test_frames)
                    gap_increase += 5  # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        return seq_id, train_frame_ids, test_frame_ids, anno 

    def _get_random_pair(self, train_dataset, test_dataset):
        is_video_dataset = train_dataset.is_video_sequence()

        min_visible_frames = self.num_train_frames
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            train_seq_id = self.rng.randint(0, train_dataset.get_num_sequences() - 1)
            train_anno, visible = train_dataset.get_sequence_info(train_seq_id)
            num_visible = np.sum(visible.astype('int64'))
            enough_visible_frames = not is_video_dataset or (
                num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            # Sample frame numbers
            train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames)
        else:
            train_frame_ids = [1] * self.num_train_frames

        is_video_dataset = test_dataset.is_video_sequence()

        min_visible_frames = self.num_test_frames
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            test_seq_id = self.rng.randint(0, test_dataset.get_num_sequences() - 1)
            test_anno, visible = test_dataset.get_sequence_info(test_seq_id)
            num_visible = np.sum(visible.astype('int64'))
            enough_visible_frames = not is_video_dataset or (
                num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            # Sample frame numbers
            test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames)
        else:
            test_frame_ids = [1] * self.num_test_frames

        return train_seq_id, test_seq_id, train_frame_ids, test_frame_ids, train_anno, test_anno

    def __iter__(self):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        neg = self.neg and self.neg > random.random()

        # Select a dataset
        if neg:
            dataset_idx = self.rng.choice(
                range(len(self.datasets)),
                p=self.p_datasets,
                replace=False)
            train_dataset = self.datasets[dataset_idx]
            
            dataset_idx = self.rng.choice(
                range(len(self.datasets)),
                p=self.p_datasets,
                replace=False)
            test_dataset = self.datasets[dataset_idx]
            train_seq_id, test_seq_id, train_frame_ids, test_frame_ids, train_anno, test_anno = \
                self._get_random_pair(train_dataset, test_dataset)

            # Get frames
            train_frames, train_anno, _ = train_dataset.get_frames(
                train_seq_id,
                train_frame_ids,
                train_anno)
            train_masks = [np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.float32)
                           for frame in train_frames]
            test_frames, test_anno, _ = test_dataset.get_frames(
                test_seq_id,
                test_frame_ids,
                test_anno)
            test_masks = [np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.float32)
                          for frame in test_frames]
        else:
            dataset_idx = self.rng.choice(
                range(len(self.datasets)),
                p=self.p_datasets,
                replace=False)
            dataset = self.datasets[dataset_idx]
            seq_id, train_frame_ids, test_frame_ids, anno = self._get_positive_pair(dataset)

            # Get frames
            if self.has_mask(dataset):
                train_frames, train_anno, train_masks, _ = dataset.get_frames_mask(
                    seq_id, train_frame_ids, anno)
                test_frames, test_anno, test_masks, _ = dataset.get_frames_mask(
                    seq_id, test_frame_ids, anno)
            else:
                train_frames, train_anno, _ = dataset.get_frames(
                    seq_id, train_frame_ids, anno)
                train_masks = [np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.float32)
                               for frame in train_frames]
                test_frames, test_anno, _ = dataset.get_frames(seq_id, test_frame_ids, anno)
                test_masks = [np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.float32)
                              for frame in test_frames]

        # Prepare data
        data = TensorDict({
            'train_images': train_frames,
            'train_anno': train_anno,
            'train_masks': train_masks,
            'test_images': test_frames,
            'test_anno': test_anno,
            'test_masks': test_masks,
            'neg': neg
        })

        # Send for processing
        yield self.processing(data, rng=self.rng)
