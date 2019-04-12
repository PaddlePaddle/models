import sys

#debug level, can be 'warn', 'verbose'
log_level = 'warn'


class KaffeError(Exception):
    pass


def print_stderr(msg):
    sys.stderr.write('%s\n' % msg)


def debug(msg):
    if log_level == 'verbose':
        print_stderr('[DEBUG]' + msg)


def notice(msg):
    print_stderr('[NOTICE]' + msg)


def warn(msg):
    print_stderr('[WARNING]' + msg)


def set_loglevel(level):
    global log_level

    if 'warn' != level and 'verbose' != level:
        raise Exception('not supported log level[%s]' % (level))

    log_level = level
