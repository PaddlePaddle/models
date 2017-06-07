""" Convert the minibatches of raw audio data within a data directory into
features that can then be used by a model. There are two modes supported
by this module: featurizing an entire epoch of data at once or doing so
piecemeal and in lockstep with the model so that a new chunk of raw data
is featurized only once the model has consumed the previous chunk.

The data directories should have been created using `data_dir_prep.py`.


useage list:
    minibatches, opts, audio_pool = _prepare(data_dir, minibatches_to_use)
    featurizer = _instantiate_featurizer(opts, audio_pool, data_dir, ce_mode)
    # we don't worry about the delay amount here, since we only care about
    # the audios. Delay only affect how the text is converted to numbers.
    _featurize(featurizer, minibatches)

 """

from __future__ import division
from __future__ import absolute_import
from builtins import range

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import copy
import logging
import os
import random
import numpy as np
from collections import namedtuple

from audio_featurizer import AudioFeaturizer
from char_map import CharMap
from libspeech import utils
#from libspeech.calculate_stats import calculate_stats_from_num_files, DTYPE
#from libspeech import schedulers

from libspeech import net

DEFAULT_DATA_SEED = 10

# use a stripped down Utterance object to store only those portions of
# an Utterance class that we need to carry out the work in preprocess_worker;
# we care because Utterances are actually quite large objects in terms of
# memory footprint, and since we've been running into memory issues during
# training (wrt swap space usage), let's play is safe.
StrippedUtterance = namedtuple('StrippedUtterance',
                               ['fname', 'duration', 'text', 'add_noise'])

logger = logging.getLogger(__name__)


def _prepare(data_dir, max_minibatches=None):
    """ Read in config data written to `data_dir` by `data_prep_dir.py` and
    produce intermediate objects required to featurize featurized data to
    the model binary.

    Args:
        :param data_dir: Filepath to data directory.
        :type data_dir: basestring
        :param max_minibatches: Maximum number of minibatches to create.
        :type max_minibatches: int or optional None

    Returns:
        minibatches (list): List of lists. Each index of the outer list
            corresponds to a list of StrippedUtterances corresponding to
            that minibatch.
        opts (dict): Configuration parameters for this folder.
        audio_pool (pool-like): A pool-like object from the
            concurrent.futures module that has a `submit` method
            to be invoked by new tasks. Also has a `shutdown` method
            that needs to be invoked when appropriate.
    """

    # Read config file
    configf = os.path.join(data_dir, 'config.json')
    assert os.path.exists(configf)
    opts = utils.read_json(configf)

    opts['char_map'] = CharMap(os.path.join(data_dir, 'chars.txt'))

    # Read num files
    numf = os.path.join(data_dir, 'num_files')
    with open(numf) as num:
        for line in num:
            total_minibatches = int(line)
    print total_minibatches

    # Go through 'list' and append utt to audio files for each file
    # Utterance is (filename, transcription, duration, add_noise)
    minibatches = [[] for _ in range(total_minibatches)]
    listfname = os.path.join(data_dir, 'list')
    with open(listfname, 'r') as listf:
        for line in listf:
            spec = line.decode('utf8').strip().split('\t')
            minibatch_id = int(spec[0])
            filename = spec[1]
            text = spec[2]
            duration = float(spec[3])
            add_noise = int(spec[4])
            utterance = StrippedUtterance(filename, duration, text, add_noise)
            minibatches[minibatch_id].append(utterance)

    if max_minibatches:
        rng = random.Random(99)
        max_minibatches = min(max_minibatches, len(minibatches))
        minibatches = rng.sample(minibatches, max_minibatches)

    # figure out the number of processes to use for feature generation.
    #max_workers = opts['data_config']['spec_parallelism']
    max_workers = 1

    logger.debug('spec_parallelism {}'.format(max_workers))

    if max_workers > 1:
        audio_pool = ProcessPoolExecutor(max_workers=max_workers)
    else:
        # Threadpoolexecutor allows us to hide some I/O
        # as it pipelines I/O with specgram calculation
        # There is also the additional benefit of having a
        # uniform interface regardless of spec_parallelism
        audio_pool = ThreadPoolExecutor(max_workers=1)
    return minibatches, opts, audio_pool


def _instantiate_featurizer(opts,
                            audio_pool,
                            data_dir,
                            cross_entropy_mode=False,
                            delay_label=0,
                            attention_mode=False):
    """ Create an AudioFeaturizer.
    Args:
        opts (dict): Output job params from `_prepare`.
        audio_pool (pool-like): thread or process pool with a `submit` method.
        data_dir (str): Path to data directory.
        cross_entropy_mode: If true, will load blank-inserted label and
                            train using cross entropy loss
        delay_label: (int, [0]): only used when cross_entropy_mode=True,
                    temporally delay the blank-inserted labels by this value.
        attention_mode: if true, add end_of_seq label to the targets.

    Returns:
        an instance of AudioFeaturizer.
    """

    return AudioFeaturizer(audio_pool, data_dir, opts['char_map'],
                           opts['data_config'], opts['rank'],
                           cross_entropy_mode, delay_label, attention_mode)


def _featurize(featurizer, minibatches, start_at=0, iteration_offset=0):
    """ Featurize all the data in `minibatches` using `featurizer`
        for consumption by the model.

    Args:
        featurizer (AudioFeaturizer): An object that can create
            features from minibatches of raw audio data.
        minibatches (iterable): Iterable of iterables. Each iterable
            is a sequence of StrippedUtterances that belong in a particular
            minibatch. This data constitutes an epoch's worth of data.
        start_at (Optional[int, 0]): Only featurize minibatches whose index
            is at least `start_at`.
        iteration_offset (int): iteration at the time of consuming
            this minibatch.
    """
    for i, minibatch in enumerate(minibatches):
        if i < start_at:
            continue
        featurizer.set_iteration(iteration_offset + i)
        featurizer.featurize_minibatch(i, minibatch)


if __name__ == '__main__':
    data_dir = './data'
    max_minibatch = 4
    minibatches, opts, audio_pool = _prepare(data_dir, max_minibatch)
    featurizer = _instantiate_featurizer(opts, audio_pool, data_dir, False)
    _featurize(featurizer, minibatches)
