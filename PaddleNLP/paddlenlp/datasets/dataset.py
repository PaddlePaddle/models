# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import collections
import io
import math
import os
import warnings

import paddle.distributed as dist
from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from typing import Iterable, Iterator, Optional, List, Any, Callable, Union

__all__ = ['MapDataset', 'DatasetReader', 'TSVDataset']


@classmethod
def get_datasets(cls, *args, **kwargs):
    """
    Get muitiple datasets like train, valid and test of current dataset.

    Example:
        .. code-block:: python

            from paddlenlp.datasets import GlueQNLI
            train_dataset, dev_dataset, test_dataset = GlueQNLI.get_datasets(['train', 'dev', 'test'])
            train_dataset, dev_dataset, test_dataset = GlueQNLI.get_datasets(mode=['train', 'dev', 'test'])
            train_dataset = GlueQNLI.get_datasets('train')
            train_dataset = GlueQNLI.get_datasets(['train'])
            train_dataset = GlueQNLI.get_datasets(mode='train')
    """
    if not args and not kwargs:
        try:
            args = cls.SPLITS.keys()
        except:
            raise AttributeError(
                'Dataset must have SPLITS attridute to use get_dataset if configs is None.'
            )

        datasets = tuple(MapDataset(cls(arg)) for arg in args)
    else:

        for arg in args:
            if not isinstance(arg, list):
                return MapDataset(cls(*args, **kwargs))
        for value in kwargs.values():
            if not isinstance(value, list):
                return MapDataset(cls(*args, **kwargs))

        num_datasets = len(args[0]) if args else len(list(kwargs.values())[0])
        datasets = tuple(
            MapDataset(
                cls(*(args[i] for args in args), **(
                    {key: value[i]
                     for key, value in kwargs.items()})))
            for i in range(num_datasets))

    return datasets if len(datasets) > 1 else datasets[0]


Dataset.get_datasets = get_datasets


class MapDataset(Dataset):
    """
    Wraps a dataset-like object as a instance of Dataset, and equips it with
    `map` and other utility methods. All non-magic methods of the raw object
    also accessible.
    Args:
        data (list|Dataset): A dataset-like object. It can be a list or a
            subclass of Dataset.
    """

    def __init__(self, data):
        self.data = data
        self._transform_pipline = []
        self.new_data = self.data

    def _transform(self, data, pipline):
        for fn in reversed(pipline):
            data = fn(data)
        return data

    def __iter__(self):
        for example in self.new_data:
            yield self._transform(
                example,
                self._transform_pipline) if self._transform_pipline else example

    def __getitem__(self, idx):
        return self._transform(
            self.new_data[idx], self._transform_pipline
        ) if self._transform_pipline else self.new_data[idx]

    def __len__(self):
        return len(self.new_data)

    def filter(self, fn):
        """
        Filters samples by the filter function and uses the filtered data to
        update this dataset.
        Args:
            fn (callable): A filter function that takes a sample as input and
                returns a boolean. Samples that return False are discarded.
        """

        self.new_data = [
            self.new_data[idx] for idx in range(len(self.new_data))
            if fn(self.new_data[idx])
        ]
        return self

    def shard(self, num_shards=None, index=None):
        """
        Use samples whose indices mod `index` equals 0 to update this dataset.
        Args:
            num_shards (int, optional): A integer representing the number of
                data shards. If None, `num_shards` would be number of trainers.
                Default: None
            index (int, optional): A integer representing the index of the
                current shard. If None, index` would be the current trainer rank
                id. Default: None.
        """
        if num_shards is None:
            num_shards = dist.get_world_size()
        if index is None:
            index = dist.get_rank()

        num_samples = int(math.ceil(len(self.new_data) * 1.0 / num_shards))
        total_size = num_samples * num_shards
        # add extra samples to make it evenly divisible
        self.new_data = [
            self.new_data[idx] for idx in range(len(self.new_data))
            if idx % num_shards == index
        ]
        if len(self.new_data) < num_samples:
            self.new_data.append(self.new_data[index + 1 - num_shards])

        return self

    def apply(self, fn, lazy=False):
        """
        Performs specific function on the dataset to transform every sample.
        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument rather than dataset.
            lazy (bool, optional): If True, transformations would be delayed and
                performed on demand. Otherwise, transforms all samples at once
                and return a new MapDataset instance. Note that if `fn` is
                stochastic, `lazy` should be True or you will get the same
                result on all epochs. Defalt: False.
        Returns:
            MapDataset: A new MapDataset instance if `lazy` is True, \
                otherwise bind `fn` as a property to transform on demand.
        """
        if lazy:
            self._transform_pipline.append(fn)
        else:
            self.new_data = [
                fn(self.new_data[idx]) for idx in range(len(self.new_data))
            ]

        return self

    def map(self, fn, lazy=False):
        """
        Performs specific function on the dataset to transform and update every sample.
        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument if lazy is True. Else it receives all examples.
            lazy (bool, optional): If True, transformations would be delayed and
                performed on demand. Otherwise, transforms all samples at once. Note that if `fn` is
                stochastic, `lazy` should be True or you will get the same
                result on all epochs. Defalt: False.
        """
        if lazy:
            self._transform_pipline.append(fn)
        else:
            self.new_data = fn(self.new_data)
        return self

    def __getattr__(self, name):
        return getattr(self.data, name)


class DatasetReader:
    """
    A base class for all DatasetReaders. It provides a `read()` function to turn 
    a data file into a MapDataset or IterDataset.

    `_get_data()` function and `_read()` function should be implemented to download
    data file and read data file into a `Iterable` of the examples.
    """

    def __init__(self, lazy: bool=False, max_examples: Optional[int]=None):
        self.lazy = lazy
        self.max_examples = max_examples

    def read_datasets(self, *args):
        datasets = []
        for arg in args:
            if os.path.exists(arg):
                datasets.append(self.read(arg))
            else:
                root = self._get_data(arg)
                datasets.append(self.read(root))

        return datasets

    def read(self, root):
        """
        Returns an dataset containing all the examples that can be read from the file path.
        If `self.lazy` is `False`, this eagerly reads all instances from `self._read()`
        and returns an `MapDataset`.
        If `self.lazy` is `True`, this returns an `IterDataset`, which internally
        relies on the generator created from `self._read()` to lazily produce examples.
        In this case your implementation of `_read()` must also be lazy
        (that is, not load all examples into memory at once).
        """
        if not isinstance(root, str):
            root = str(root)

        if self.lazy:
            example_iter = self._read(root)

            label_list = self.get_labels()

            if label_list is not None:

                label_dict = {}
                for i, label in enumerate(label_list):
                    label_dict[label] = i

                def generate_examples(example_iter):
                    for example in example_iter:
                        if 'labels' not in example.keys():
                            raise ValueError(
                                "Keyword 'labels' should be in example if get_label() is specified."
                            )
                        else:
                            for label_idx in range(len(example['labels'])):
                                example['labels'][label_idx] = label_dict[
                                    example['labels'][label_idx]]

                            yield example

                return MapDataset(generate_examples(example_iter))
            else:
                return MapDataset(example_iter)

        else:
            examples = self._read(root)

            # Then some validation.
            if not isinstance(examples, list):
                examples = list(examples)

            if not examples:
                raise ValueError(
                    "No instances were read from the given filepath {}. "
                    "Is the path correct?".format(root))

            label_list = self.get_labels()

            # Convert class label to label ids.
            if label_list is not None:
                if 'labels' not in examples[0].keys():
                    raise ValueError(
                        "Keyword 'labels' should be in example if get_label() is specified."
                    )

                label_dict = {}
                for i, label in enumerate(label_list):
                    label_dict[label] = i

                for idx in range(len(examples)):
                    for label_idx in range(len(examples[idx]['labels'])):
                        examples[idx]['labels'][label_idx] = label_dict[
                            examples[idx]['labels'][label_idx]]

            return MapDataset(examples)

    def _read(self, file_path: str):
        """
        Reads examples from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        """
        raise NotImplementedError

    def _get_data(self, mode: str):
        """
        Download examples from the given URL and customized split informations and returns a filepath.
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Return list of class labels of the dataset if specified.
        """
        return None


class TSVDataset(Dataset):
    """
    Common tab separated text dataset that reads text fields based on provided
    sample splitter and field separator.
    The returned dataset includes samples, each of which can either be a list
    of text fields if field_separator is specified, or otherwise a single
    string segment produced by the sample_splitter.
    Args:
        filename (str|list of str): Path to the input text file or list of
            paths to the input text files.
        encoding (str): File encoding format. Default: 'utf8'.
        sample_splitter (function): A function that splits the dataset string
            into samples. Default: str.splitlines
        field_separator (function|None): A function that splits each sample
            string into list of text fields. If None, raw samples are returned
            according to `sample_splitter`. Default: split method of str with
            tab as separator.
        num_discard_samples (int): Number of samples discarded at the head of
            the first file. Default: 0.
        field_indices (list|int|None): If set, for each sample, only fields
            with provided indices are selected as the output. Otherwise all
            fields are returned. Default: None.
        allow_missing (bool): If set to True, no exception will be thrown if
            the number of fields is smaller than the maximum field index
            provided.  Default: False.
        
    Example:
        assume `test.tsv` contains the following content:
        Id\tFirstName\tLastName
        a\tmale\tTom
        b\tFemal\tCat
        discard the first line and select the 0th and 2nd fields
        .. code-block:: python

            from paddlenlp.datasets import TSVDataset
            dataset = TSVDataset('test.tsv', num_discard_samples=1,
                                field_indices=[0, 2])
            dataset[0] # ['a', 'Tom']
            dataset[1] # ['b', 'Cat']
    """

    def __init__(self,
                 filename,
                 encoding='utf-8',
                 sample_splitter=lambda x: x.splitlines(),
                 field_separator=lambda x: x.split('\t'),
                 num_discard_samples=0,
                 field_indices=None,
                 allow_missing=False):
        assert sample_splitter, 'sample_splitter must be specified.'

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._sample_splitter = sample_splitter
        self._field_separator = field_separator
        self._num_discard_samples = num_discard_samples
        self._field_indices = field_indices
        self._allow_missing = allow_missing
        self.data = self._read()

    def _should_discard(self):
        discard = self._num_discard_samples > 0
        self._num_discard_samples -= 1
        return discard

    def _field_selector(self, fields):
        if not self._field_indices:
            return fields
        try:
            result = [fields[i] for i in self._field_indices]
        except IndexError as e:
            raise (IndexError('%s. Fields = %s' % (str(e), str(fields))))
        return result

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            samples = (s for s in self._sample_splitter(content)
                       if not self._should_discard())
            if self._field_separator:
                if not self._allow_missing:
                    samples = [
                        self._field_selector(self._field_separator(s))
                        for s in samples
                    ]
                else:
                    selected_samples = []
                    num_missing = 0
                    for s in samples:
                        try:
                            fields = self._field_separator(s)
                            selected_samples.append(
                                self._field_selector(fields))
                        except IndexError:
                            num_missing += 1
                    if num_missing > 0:
                        warnings.warn('%d incomplete samples in %s' %
                                      (num_missing, filename))
                    samples = selected_samples
            all_samples += samples
        return all_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
