"""
SimNet config
"""

import six
import json


class SimNetConfig(object):
    """
    simnet Config
    """

    def __init__(self, args):
        self.task_mode = args.task_mode
        self.config_path = args.config_path
        self._config_dict = self._parse(args.config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing simnet model config file '%s'" % config_path)

        else:
            if config_dict["task_mode"] != self.task_mode:
                raise ValueError(
                    "the config '{}' does not match the task_mode '{}'".format(self.config_path, self.task_mode))
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        """
        Print Config
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')
