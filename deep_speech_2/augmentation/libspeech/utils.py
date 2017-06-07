''' General purpose utility classes
'''
from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object

import abc
import bisect
import datetime
import errno
import json
import logging
import operator
import os
import random
import re
import string
import sys
import time

from concurrent.futures import ProcessPoolExecutor
from future.utils import with_metaclass

logger = logging.getLogger(__name__)


class TextCleaner(object):
    allowed_chars = set(list(string.ascii_letters.lower()) + list(" '"))

    remove_chars = '=#*:!",.;?()_\[\]/'
    remove_subs = [
        "\xc3\xa6", "\xef\xbb\xbf", "\xc3\xb1", "\xc3\xa8", "\xc3\xaf",
        "\xc3\x86", "\xc3\xb6", "\xc3\xa9", "\xc3\xab", "\xc3\x8f"
    ]
    space_subs = ["--", "-", "' '", "' ", " '"]

    escape_rewrites = dict([(c, '') for c in remove_chars] + [
        (c, '') for c in remove_subs
    ] + [(c, " ") for c in space_subs] + [('&', ' and '), ("\xc3\xb4", 'o'), (
        "\xc3\xa2", 'a'), ("\xc3\xaa", 'e'), ("`", "'")])

    # for matching \' at beginning or end of sentence
    no_escape_rewrites = {"^'": "", "'$": ""}
    patterns = (list(map(re.escape, escape_rewrites.keys())) +
                list(no_escape_rewrites.keys()))
    cleaning_regex = re.compile("(%s)" % "|".join(patterns))

    # for matching \' at beginning or end of sentence
    escape_rewrites["'"] = ""

    @classmethod
    def clean(cls, line):
        ''' Takes a line and scrubs it
        '''

        def replace_fn(x):
            return cls.escape_rewrites[x.string[x.start():x.end()]]

        line = cls.cleaning_regex.sub(replace_fn, line).lower().strip()
        # Do it a second time to catch newly exposed start \' and space \'
        line = cls.cleaning_regex.sub(replace_fn, line).lower().strip()
        is_valid = set(list(line)).issubset(cls.allowed_chars)
        return (is_valid, line)


def now_str():
    return datetime.datetime.strftime(datetime.datetime.now(),
                                      '%Y-%m-%d %H:%M:%S')


def get_lines(filename):
    num_lines = 0
    with open(filename) as f:
        for line in f:
            num_lines += 1
    return num_lines


def tuple_add(tup1, tup2):
    ''' Reduction function that takes (a,b,...) and (c,d,...)
        and returns (a+c, b+d, ....)
    '''
    return list(map(operator.add, tup1, tup2))


def truncate_float(val, ndigits=6):
    """ Truncates a floating-point value to have the desired number of
    digits after the decimal point.

    Args:
        val (float): input value.
        ndigits (int): desired number of digits.

    Returns:
        truncated value
    """
    p = 10.0**ndigits
    return float(int(val * p)) / p


def list_dict_update(dict1, dict2):
    ''' addes dict2 to dict1 by concatenating lists
    '''
    for key in dict2.keys():
        if key in dict1 and hasattr(dict1, '__len__'):
            dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]


def timed(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        func(*arg, **kw)
        t2 = time.time()
        print('%s took %0.2f s' % (func.__name__, (t2 - t1)))

    return wrapper


def chunk_list(dataset, chunk_size, rank=0, num_splits=1, truncate=True):
    ''' Splits an array \in dataset into smaller lists all of them exactly \in
        chunk_size long. If \in rank and \in num_splits are specified, only
        returns a slice of each chunk corresponding to the \in rank
    '''
    N = len(dataset)
    slice_size = int(chunk_size // num_splits)
    if N == 0 or slice_size == 0:
        return []
    num_slices = int(int(N // slice_size) // num_splits) * num_splits
    assert slice_size * num_splits == chunk_size, 'Slice equally among ranks'
    assert num_slices % num_splits == 0, 'Not equal chunks'
    assert rank < num_splits, 'Rank should never be greater than num_splits'
    logger.info("{} slices of size {} {}".format(num_slices, slice_size, N))
    sys.stdout.flush()
    if slice_size * num_slices != N and truncate:
        N = slice_size * num_slices
    return [
        dataset[index:index + slice_size]
        for index in range(slice_size * rank, N, chunk_size)
    ]


def create_file(fname):
    open(fname, 'w').close()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parmap(func, items):
    ''' Parallel map function, use like basic map call
    '''
    with ProcessPoolExecutor() as executor:
        res = [r for r in executor.map(func, items) if r]
    return res


def path_join(*parts):
    return os.path.join(*[str(p) for p in parts])


def expand_path(path):
    return os.path.expandvars(os.path.expanduser(path))


def expand_vals_inplace(D):
    """Expand environment variables in all strings in dict-of-dicts.

    Args:
        D (dict): dictionary containing strings to expand

    Note:
        This is done in-place and does not expand anything contained within
        lists.
    """
    for key, val in D.items():
        if isinstance(val, dict):
            expand_vals_inplace(val)
        elif isinstance(val, basestring):
            D[key] = os.path.expandvars(val)


def write_json(json_data, filepath):
    """ Write out a dictionary into the filepath.

    Args:
        json_data (dict): The dictionary to write out.
        filepath (str): The filename of the new json file.

    Returns:
        A dictionary of the values in the json file.
    """
    with open(filepath, 'w') as fp:
        json.dump(json_data, fp)


def read_json(configf):
    """ Read in a json file and turn it into a dictionary.

    Args:
        configf (str): The filename of a json file.

    Returns:
        A dictionary of the values in the json file.
    """
    with open(configf) as config_file:
        config = json.load(config_file)
    expand_vals_inplace(config)

    return config


def convert_pickle_to_json(pickle_file, json_file):
    """Convert pickle file to json file.

    Args:
        pickle_file (str): path to pickle file to convert
        json_file (str): path to output json file

    Note: If contents of pickle are not serializable via JSON,
        an exception will be raised.
    """
    import pickle
    import json

    with open(pickle_file, 'r') as fp:
        data = pickle.load(fp)

    with open(json_file, 'w') as fp:
        json.dump(data, fp)


def get_first_smaller(items, value):
    index = bisect.bisect_left(items, value) - 1
    assert items[index] < value, \
        'get_first_smaller failed! %d %d' % (items[index], value)
    return items[index]


def get_first_larger(items, value):
    'Find leftmost value greater than value'
    index = bisect.bisect_right(items, value)
    assert index < len(items), \
        "no noise bin exists for this audio length (%f)" % value
    assert items[index] > value, \
        'get_first_larger failed! %d %d' % (items[index], value)
    return items[index]


def make_bandwidth_string(read_size, read_time):
    '''Return formated string bytes/second
    '''
    suff = ['B', 'K', 'M', 'G', 'T']
    ind, bandwidth = 0, 100

    if read_time == 0:
        return '+inf/s'
    else:
        while True:
            bandwidth = read_size / (read_time * (1024**ind))
            if bandwidth < 1024:
                break
            ind += 1
        return '%.0f%s/s' % (bandwidth, suff[ind])


class ReservoirSampler(object):
    def __init__(self, num_samples=1):
        self.samples = []
        self.num_samples = num_samples
        self.num_observations = 0
        self.rng = random.Random(10)

    def observe(self, sample):
        if len(self.samples) < self.num_samples:
            assert self.num_observations < self.num_samples
            self.samples.append(sample)
        else:
            r = self.rng.randint(0, self.num_observations)
            if r < self.num_samples:
                self.samples[r] = sample
        self.num_observations += 1

    def __getitem__(self, ind):
        return self.samples[ind]


class FileTemplate(object):
    """ Many files we produce in speech-dl follow the template
    {name}{identifier}{extension}. This class makes it easy to make
    a particular instance of a particular template.
    """

    template = '{name}{identifier}{extension}'

    def __init__(self, write_dir, name, extension=''):
        self.write_dir = write_dir
        self.name = name
        self.extension = extension

    def instantiate(self, identifier):
        """ Create an instance of this template using this
        particular identifier.

        Args:
            identifier (object): The identifier to use.

        Returns:
            filename (str): A fully-qualified path to the
                instantiated template.
        """
        filename = self.template.format(
            name=self.name, identifier=identifier, extension=self.extension)
        filename = os.path.join(self.write_dir, filename)
        return filename


class Featurizer(with_metaclass(abc.ABCMeta, object)):
    """ An object that will produce minibatches of data for consumption by
        another entity. Also is in charge of removing the data once it's been
        consumed. """

    @abc.abstractmethod
    def clear_minibatch(self, minibatch_id):
        """ Remove all files related to a minibatch.

        Args:
            minibatch_id (int): minibatch identifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def featurize_minibatch(self, minibatch_id, minibatch):
        """ Featurize the data in `minibatch` and write it out.
        Args:
            minibatch_id (int): minibatch identifier.
            minibatch (iterable): Iterable of samples to be featurized.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def write_dir(self):
        """ Returns the directory where featurized data should be written. """
        raise NotImplementedError


class Orchestrator(object):
    """ An object that manipulates an instance of `Featurizer` to provide
    batched features to a binary as well as remove them once the data has been
    consumed. Specifically, this class creates sentinel files the binary
    deletes when the input has been consumed.

    Note: It is crucial that the binary and `Orchestrator`
    agree on what the sentinel files should look like.

    Args:
        feature_manipulator (Featurizer): An instance of
            `Featurizer` to control feature file generation.
        buffer_size (Optional[int, 0]): The number of minibatches-worth of
            files that we are allowed to generate ahead of what the
            binaries have consumed.
        test_mode (Optional[boolean, False]): If False, this class waits for
            someone to consume its sentinel files. This is useful to turn off
            during unit testing.
        binary_proc (Optional[subprocess.Popen, None]): The process handler
            for the model binary reading from this data directory. Used by the
            Orchestrator to determine if the model will continue consuming
            files.
    """

    def __init__(self,
                 feature_manipulator,
                 buffer_size=0,
                 test_mode=False,
                 binary_proc=None):
        self.featurizer = feature_manipulator
        self.sentinel_template = \
            FileTemplate(self.featurizer.write_dir, 'sentinel')
        self.buffer_size = buffer_size if buffer_size > 0 else sys.maxsize
        self.live_ids = []
        self.test_mode = test_mode
        self.binary_proc = binary_proc

    def _sentinels(self, mb_id):
        """ Create sentinel files for this minibatch id.
        Args:
            mb_id (int): Minibatch identifier.
        Returns:
            sentinel_files (iterable): List of sentinel files for this
                minibatch.
        """
        sentinel_files = []
        identifier = '{}_0'.format(mb_id)
        sentinel_file = self.sentinel_template.instantiate(identifier)
        sentinel_files.append(sentinel_file)
        return sentinel_files

    def _is_alive(self, mb_id):
        """ Check if any sentinel file still exists for the minibatch `mb_id`.
        Args:
            mb_id (int): Minibatch id.

        Returns:
            boolean indicating whether any sentinel file for minibatch `mb_id`
                hasn't been consumed yet.
        """
        status = []
        for sentinel in self._sentinels(mb_id):
            status.append(os.path.exists(sentinel))
        return any(status)

    def _flush_dead(self):
        """ Removes feature files for consumed minibatches. """
        dead = [iid for iid in self.live_ids if not self._is_alive(iid)]
        for iid in dead:
            if self.buffer_size != sys.maxsize:
                self.featurizer.clear_minibatch(iid)
            self.live_ids.remove(iid)

    def register_live(self, mb_id):
        """ Create every sentinel file for minibatch `mb_id`.

        Args:
            mb_id (int): Minibatch id.

        """
        self.live_ids.append(mb_id)
        for sentinel in self._sentinels(mb_id):
            create_file(sentinel)

    def binary_alive(self):
        """ Check if the model binary is still alive.

        Returns:
            True if binary process wasn't specified or if it's alive.
            False otherwise.
        """
        if not self.binary_proc:
            return True
        elif self.binary_proc.poll() is None:
            return True
        else:
            return False

    def apply_on(self, minibatches, start_at_index=0, iteration_offset=0):
        """ Feed the binary each minibatch in `minibatches`. This involves
        creating all file system state for each minibatch, setting up all
        sentinel files for the minibatch, and then cleaning up once the binary
        has consumed the input.

        Args:
            minibatches (iterable): Sequence of minibatches, each of which
                is an iterable of data of a form `self.featurizer` can process.
            start_at_index (Optional[int, 0]): Minibatch index to start at.
                Any minibatch before start_at_index will be skipped.
        """
        for ind, minibatch in enumerate(minibatches):
            if ind < start_at_index:
                continue
            if not self.binary_alive():
                break

            while self.binary_alive() and \
                    len(self.live_ids) >= self.buffer_size:
                self._flush_dead()

            self.featurizer.set_iteration(iteration_offset + ind)
            self.featurizer.featurize_minibatch(ind, minibatch)
            self.register_live(ind)

        if not self.test_mode:
            while self.binary_alive() and len(self.live_ids) > 0:
                self._flush_dead()


def configure_logging(console_log_level=None,
                      console_log_format=None,
                      file_log_path=None,
                      file_log_level=None,
                      file_log_format=None,
                      clear_handlers=False):
    """Setup logging.

    This configures either a console handler, a file handler, or both and
    adds them to the root logger.

    Args:
        console_log_level (logging level): logging level for console logger
        console_log_format (str): log format string for console logger
        file_log_path (str): full filepath for file logger output
        file_log_level (logging level): logging level for file logger
        file_log_format (str): log format string for file logger
        clear_handlers (bool): clear existing handlers from the root logger

    Note:
        A logging level of `None` will disable the handler.
    """
    if file_log_format is None:
        file_log_format = \
            '%(asctime)s %(levelname)-8s (%(name)-12s) %(message)s'

    if console_log_format is None:
        console_log_format = \
            '%(asctime)s %(levelname)-8s (%(name)-12s) %(message)s'

    # configure root logger level
    rootLogger = logging.getLogger()
    root_level = rootLogger.level
    if console_log_level is not None:
        root_level = min(console_log_level, root_level)
    if file_log_level is not None:
        root_level = min(file_log_level, root_level)
    rootLogger.setLevel(root_level)

    # clear existing handlers
    if clear_handlers and len(rootLogger.handlers) > 0:
        print("Clearing {} handlers from root logger."
              .format(len(rootLogger.handlers)))
        for handler in rootLogger.handlers:
            rootLogger.removeHandler(handler)

    # file logger
    if file_log_path is not None and file_log_level is not None:
        log_dir = os.path.dirname(os.path.abspath(file_log_path))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(file_log_path)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        rootLogger.addHandler(file_handler)

    # console logger
    if console_log_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(logging.Formatter(console_log_format))
        rootLogger.addHandler(console_handler)


def get_label_from_alignment(alignment):
    """ Alignment is a string of characters in the charmap (plus <BLANK>).
    It can be reduced to a labeling by removing the <BLANK>(s),
    and collapsing consecutively repeated characters.
    """
    aln = alignment.replace("<BLANK>", "_")
    label = []
    c_prev = "INI"
    for c in aln:
        if c == c_prev:
            continue
        if c != "_":
            label.append(c)
        c_prev = c
    label = ''.join(label)
    return label
