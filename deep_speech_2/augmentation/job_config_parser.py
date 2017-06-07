""" A module that parses a JSON input and creates a dictionary of options
    (as well as default values) that should be used for training a model.
"""
import os

from libspeech import utils


def parse_json(config_file,
               network_config='',
               data_dir=None,
               model_dir=None,
               rank=None):
    """ Set up all job-related and parameters for this run.

    Args:
        :param config_file: json file to parse for variables.
        :type config_file:  basestring
        :param data_dir: Provide a custom directory
            for data directory creation.
        :type data_dir: [str, None], Optional
        :param model_dir: Provide a custom directory
            for model directory creation.
        :type model_dir: [str, None], Optional
        :param rank: Provide a custom rank to use for this process;
            the provided by MPI (if any) is ignored.
        :type rank: int, Optional

    Returns:
        opts (dict): dictonary of job-related parameters.
    """
    opts = utils.read_json(config_file)
    return opts
