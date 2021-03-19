import csv
import io
import os
from typing import List, Dict, Tuple, Optional, Union

import paddle
import numpy as np

import utils as U


class InputExample(object):
    """
    Input example of one audio sample.
    """

    def __init__(self,
                 guid: int,
                 source: Union[list, str],
                 label: Optional[str]=None):
        self.guid = guid
        self.source = source
        self.label = label


class BaseSpeechDataset(object):
    """
    Base class of speech dataset.
    """

    def __init__(self,
                 base_path: str,
                 mode: Optional[str]="train",
                 data_file: Optional[str]=None):
        """
        Ags:
            base_path (:obj:`str`): The directory to the whole dataset.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).
            data_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The data file name, which is relative to the base_path.
        """
        self.data_file = os.path.join(base_path, data_file)
        self.mode = mode

    def _load_label_data(self):
        """
        Loads labels from label file.
        """
        raise NotImplementedError

    def _read_file(self, input_file: str):
        """
        Reads data from file.
        Args:
            input_file (:obj:`str`) : The data file to be read.
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Gets all labels.
        """
        raise NotImplementedError


class AudioClassificationDataset(BaseSpeechDataset, paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """
    supported_features = ['raw', 'mfcc', 'fbanks']

    def __init__(self,
                 base_path: str,
                 mode: str='train',
                 data_file: str=None,
                 data_type: str='npz',
                 feature_type: str='raw'):
        super(AudioClassificationDataset, self).__init__(
            base_path=base_path, mode=mode, data_file=data_file)

        self.feature_type = feature_type
        self.data_type = data_type

        self.examples = self._read_file(self.data_file)
        self.records = self._convert_examples_to_records(self.examples)

    def _convert_examples_to_records(
            self, examples: List[InputExample]) -> List[dict]:
        """
        Converts all examples to records which the model needs.
        Args:
            examples(obj:`List[InputExample]`): All data examples returned by _read_file.
        Returns:
            records(:obj:`List[dict]`): All records which the model needs.
        """
        records = []
        for example in examples:
            record = {}
            if self.feature_type == 'raw':
                record['feat'] = example.source
                record['label'] = example.label
            elif self.feature_type == 'mfcc':
                # TODO: convert wave to mfcc features
                raise NotImplementedError(f'MFCC not supported.')
            elif self.feature_type == 'fbanks':
                # TODO: convert wave to fbanks features
                raise NotImplementedError(f'FBanks not supported.')
            else:
                raise RuntimeError(\
                    f"Unknown type of self.feature_type: {self.feature_type}, it must be one in {self.supported_features}")

            records.append(record)
        return records

    def __getitem__(self, idx):
        """
        Overload this method when doing extra feature processes or data augmentation. 
        """
        record = self.records[idx]
        return np.array(record['feat']), np.array(
            record['label'], dtype=np.int64)

    def __len__(self):
        return len(self.records)


class ESC50(AudioClassificationDataset):
    sr = 44100  # sample rate
    input_length = 66650  # input length
    n_class = 50  # class num
    n_crops = 10  # dev audio crops num

    # TODO: replace data_path with a env DATA_HOME
    base_path = '/ssd3/chenxiaojie06/envnet/dataset/esc50'

    def __init__(self,
                 mode: str='train',
                 data_file: str='wav44.npz',
                 data_type: str='npz',
                 feature_type: str='raw',
                 soft_label: bool=True,
                 bc_learning: bool=True):

        super(ESC50, self).__init__(
            base_path=self.base_path,
            mode=mode,
            data_file=data_file,
            data_type=data_type,
            feature_type=feature_type)

        self.soft_label = soft_label
        self.bc = bc_learning
        self.preprocess_funcs = self._preprocess_setup()

    def _read_file(self, input_file: str, split=3) -> List[InputExample]:
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))

        examples = []
        if self.data_type == 'npz':
            dataset = np.load(os.path.join(self.data_file), allow_pickle=True)
            audio_id = 0
            for fold_tag, data in dataset.items():
                if self.mode == 'train' and fold_tag == f'fold{split}':
                    continue
                if self.mode != 'train' and fold_tag != f'fold{split}':
                    continue

                sounds = data.item()['sounds']
                labels = data.item()['labels']
                for sound, label in zip(sounds, labels):
                    example = InputExample(
                        guid=audio_id, source=sound, label=label)
                    audio_id += 1
                    examples.append(example)
        elif self.data_type == 'wav_scp':
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                audio_id = 0
                for line in reader:
                    example = InputExample(
                        guid=audio_id, source=line[0],
                        label=line[1])  # wavfile \t label
                    audio_id += 1
                    examples.append(example)
        else:
            raise NotImplementedError(
                f'Only soppurts npz|wav_scp data type, but got {self.data_type}')

        return examples

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.bc:  # bc learning, mix audio
                # Select two training examples
                while True:
                    record_a = np.random.choice(self.records)
                    record_b = np.random.choice(self.records)
                    sound1, label1 = record_a['feat'], record_a['label']
                    sound2, label2 = record_b['feat'], record_b['label']
                    if label1 != label2:
                        break

                sound1 = self._preprocess(sound1)
                sound2 = self._preprocess(sound2)

                r = np.random.random()
                sound = U.mix(sound1, sound2, r, self.sr).astype(np.float32)
                sound = U.random_gain(6)(sound).astype(np.float32)

                assert self.soft_label, "BC learning only work in soft label mode."
                eye = np.eye(self.n_class)
                label = (eye[label1] * r + eye[label2] *
                         (1 - r)).astype(np.float32)
            else:
                record = self.records[idx]
                sound, target = record['feat'], record['label']
                sound = self._preprocess(sound).astype(np.float32)
                sound = U.random_gain(6)(sound).astype(np.float32)

                if self.soft_label:
                    label = np.eye(self.n_class)[int(target)].astype(np.float32)
                else:
                    label = int(target)
        else:
            record = self.records[idx // self.n_crops]
            sound, target = record['feat'], record['label']
            sound = self._preprocess(sound)[idx %
                                            self.n_crops].astype(np.float32)

            if self.soft_label:
                label = np.eye(self.n_class)[int(target)].astype(np.float32)
            else:
                label = int(target)

        return sound.reshape(1, 1, -1), label

    def _preprocess_setup(self):
        """
        Setup preprocess for audio data.
        """
        if self.mode == 'train':
            funcs = []

            funcs += [U.random_scale(1.25)]
            funcs += [
                U.padding(self.input_length // 2),
                U.random_crop(self.input_length),
                U.normalize(32768.0),
            ]
        else:
            funcs = [
                U.padding(self.input_length // 2),
                U.normalize(32768.0),
                U.multi_crop(self.input_length, self.n_crops),
            ]
        return funcs

    def _preprocess(self, sound):
        """
        Execute preprocess for audio data.
        """
        for f in self.preprocess_funcs:
            sound = f(sound)
        return sound

    def __len__(self):
        if self.mode == 'train':
            return len(self.records)
        else:
            return len(self.records) * self.n_crops


if __name__ == "__main__":
    train_dataset = ESC50(mode='train')
    dev_dataset = ESC50(mode='dev')
