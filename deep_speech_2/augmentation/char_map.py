""" 
Build char-map from datasets
"""

from __future__ import print_function
from builtins import str
from past.builtins import basestring
import os
import logging

CHAR_MAP_FILE = 'chars.txt'
SPACE = "<SPACE>"
BLANK = "<BLANK>"
END_OF_SEQ = "eos"

logger = logging.getLogger(__name__)


class CharMap(dict):
    """
    Char map maps from an alphabet to an index, required for interpreting
       Model output
    """

    def __init__(self, inp, attention_mode=False):
        """
        Args:
            :param inp:
                If path, loads chars.txt from disk
                If dict, uses it to init
                If set, sorts the items and indexes them
            :type inp: {path, dict, set}
            :param attention_mode: Used to add END_OF_SEQ symbol to
                            char_map in attention training.
            :type attention_mode: boolean
        """
        super(CharMap, self).__init__()
        if isinstance(inp, basestring):
            if os.path.exists(inp):
                self.update(CharMap.deserialize(inp))
            else:
                raise Exception("Filepath given ({}), but it doesn't exist!"
                                .format(inp))
        elif type(inp) is dict:
            self.update(inp)
        elif type(inp) is set:
            for index, char in enumerate(sorted(inp)):
                self[char] = str(index + 1)
        else:
            raise Exception("CharMap can't handle input {} of type {}!"
                            .format(inp, type(inp)))
        if attention_mode:
            self[END_OF_SEQ] = str(len(self) + 1)

    @classmethod
    def deserialize(cls, char_map_file):
        """
        Load char map from disk

        Args:
            :param char_map_file: path to chars.txt file
            :type char_map_file: basestring
        """
        if os.path.isdir(char_map_file):
            char_map_file = os.path.join(char_map_file, CHAR_MAP_FILE)
        logger.info("Loading char map from {}".format(char_map_file))
        with open(char_map_file, 'r') as fptr:
            char_maps = [line.decode('utf-8') for line in fptr.readlines()]
            return dict(char_map.strip().split() for char_map in char_maps)

    def serialize(self, save_dir):
        """Save the char map to disk

        Args:
            :param save_dir: location where chars.txt will be written
                This location must already exist
            :type save_dir: basestring
        """
        char_map_file = os.path.join(save_dir, CHAR_MAP_FILE)
        with open(char_map_file, 'w') as fptr:
            for char, index in sorted(self.items()):
                map_str = char + ' ' + index + '\n'
                fptr.write(map_str.encode('utf8'))

    def apply(self,
              transcript,
              cross_entropy_mode,
              delay_label,
              attention_mode=False):
        """
        Produces token mapping for one text transcription.

        Args:
            :param transcript: Raw text to converted. All characters must be
            represented in the char map
            :type transcript:  basestring

            :param cross_entropy_mode: If true, will convert blank inserted
            labeling into a list of ints. The length of the list is same as
            the number of time frames.
            :type cross_entropy_mode: boolean

            :param attention_mode: If true, will add end_of_seq at the end of each
            label.
            :type attention_mode: boolean

            :param delay_label: temporally delay the text label for certain steps.
            :type delay_label: int
        Note:
            Inserts space between words and after all words in transcript,
            except for empty strings

        """
        if cross_entropy_mode:
            try:
                label = []
                for t in transcript:
                    if t == " ":
                        label.append(self[SPACE])
                    elif t == "_":  # for <BLANK>
                        label.append('0')
                    else:
                        label.append(self[t])

                if delay_label:  # this can only happen in CrossEntropy mode
                    label = ['0'] * delay_label + label[:-delay_label]
                return label
            except KeyError as e:
                raise Exception('Key {0} not in char map'.format(e))
        else:
            try:
                label = []
                for word in transcript.split():
                    label += [self[t] for t in list(word)]
                    if SPACE in self:
                        label.append(self[SPACE])
                if attention_mode:
                    label.append(self[END_OF_SEQ])
                return label
            except KeyError as e:
                raise Exception('Key {0} not in char map'.format(e))

    @property
    def cardinality(self):
        """
        Size of the alphabet as stored in the char map
        """
        return 1 + len(self)


"""
def create_char_map_for_dataset(config_fname, save_dir=None):
    #
    Standalone function to calculate the union of all charsets
    from multiple datasets and save it to disk

    Args:
        :param config_fname: path to config file. It is in the same
            format as the train configs, but only the data_config subsection
            is required and used.
        :type config_fname: basestring
    #

    from libspeech.run_model.data_dir_prep import get_dataset_info
    from libspeech.run_model.job_config_parser import parse_json

    opts = parse_json(config_fname, save_dir)
    _, _, char_map, _ = get_dataset_info(opts)
    print('Saving chars.txt to {}'.format(save_dir))
    char_map.serialize(save_dir)
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_config",
        type=str,
        metavar="job_config",
        help="Training config with data_sources specified")
    parser.add_argument("save_dir", type=str, help="Path to save the char_map")
    args = parser.parse_args()
    #create_char_map_for_dataset(args.job_config, args.save_dir)
