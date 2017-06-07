import logging
import os
import sys
import numpy as np

_local = False
logger = logging.getLogger(__name__)
_cache = {}


def _warn(name, value):
    # logger.warn("%s: %s" % (name, value))
    pass


def _error(name, value):
    logger.error("%s: %s" % (name, value))
    pass


FILECLIENT = os.path.realpath("/tools/fileserver/20160609-e8b0ef0/python/")
sys.path.append(FILECLIENT)
try:
    import fileclient
except Exception as ex:
    _error("fileclient module can not be imported. Switch to local mode",
           FILECLIENT)
    _error("import error", ex.message)
    _local = True


def _file_content(file_pointer):
    return np.array(file_pointer, copy=False).tostring()


def _read_file_over_network(url):
    local_filename = get_local_url(url)
    file = fileclient.get_file(local_filename)
    return _file_content(file)


def _read_file_locally(url):
    local_filename = get_local_url(url)
    with open(local_filename, "r") as fp:
        value = fp.read()
    return value


def check_net_url(url):
    return url.startswith("net:")


def get_local_url(url):
    if check_net_url(url):
        return url[4:]
    return url


def read_file_async(filename):
    global _local
    if filename in _cache:
        return
    _warn("read_async", filename)
    value = _read_file_locally(filename) if _local else \
        _read_file_over_network(filename)
    _cache[filename] = value


def read_files_async(filenames):
    _warn("read_async_files", filenames)
    for filename in filenames:
        if check_net_url(filename):
            read_file_async(filename)


def read_file_sync(filename):
    read_file_async(filename)
    return pop_object(filename)


def read_files_sync(filenames):
    read_files_async(filenames)
    return [pop_object(filename) for filename in filenames]


def init(opts):
    global _local
    if not _local:
        if 'ionode' in opts:
            _server = opts['ionode']
            fileclient.set_servers([_server])
        else:
            _error("No 'ionode' address is set. Switch to local mode", "")
            _local = True


def done(*args, **kwargs):
    pass


def _add_object(key, value):
    if key not in _cache:
        _cache[key] = value
    _error("Key is already cached", key)


def _del_object(key):
    if key in _cache:
        del _cache[key]
    else:
        _error("Key is not found", key)


def get_object(key):
    if key in _cache:
        return _cache[key]
    else:
        _error("Key is not found, reading file synchronously", key)
        read_file_sync(key)


def pop_object(key):
    value = get_object(key)
    _del_object(key)
    return value


def shuffle_happened(new_index):
    """
    The goal of net.shuffle_happened is to let net module know about the order
    of the samples, even in epoch 0. currently it is just a placeholder.

    Args:
        :param new_index: the new shuffeled batch file
        :type new_index: list

    Returns:
        Nothing.
    """
    _warn("Data has been shuffled. Number of indexes", len(new_index))
