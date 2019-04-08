"""
EmoTect config
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json

class EmoTectConfig(object):
    """
    EmoTect Config
    """
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing emotect model config file '%s'" % config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        """
        Print Config
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')
