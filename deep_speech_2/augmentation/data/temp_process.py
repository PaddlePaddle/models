import os
import random


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
        :param rank: Provide a custom rank to use for this process
            the provided by MPI (if any) is ignored.
        :type rank: int, Optional

    Returns:
        opts (dict): dictonary of job-related parameters.
    """
    with open(config_file) as lines:
        for line in lines:
            line_info = eval(line)
            new_array = []
            id = random.randint(0, 2)
            fname = line_info["key"]
            text = line_info["text"]
            duration = line_info["duration"]
            add_noise = '1'
            new_array = [str(id), fname, text, str(duration), add_noise]
            print '\t'.join(new_array)


if __name__ == '__main__':
    config_file = './list_c'
    parse_json(config_file)
