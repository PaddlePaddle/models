import sys


class KaffeError(Exception):
    pass


def print_stderr(msg):
    sys.stderr.write('%s\n' % msg)
