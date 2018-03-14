from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from six import reraise
from tblib import Traceback
from multiprocessing import Manager, Process
import posix_ipc, mmap

import numpy as np


def to_lodtensor(data, place):
    """convert tensor to lodtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = numpy.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def lodtensor_to_ndarray(lod_tensor):
    """conver lodtensor to ndarray
    """
    dims = lod_tensor.get_dims()
    ret = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ret.ravel()[i] = lod_tensor.get_float_element(i)
    return ret, lod_tensor.lod()


def batch_to_ndarray(batch_samples, lod):
    frame_dim = batch_samples[0][0].shape[1]
    batch_feature = np.zeros((lod[-1], frame_dim), dtype="float32")
    batch_label = np.zeros((lod[-1], 1), dtype="int64")
    start = 0
    for sample in batch_samples:
        frame_num = sample[0].shape[0]
        batch_feature[start:start + frame_num, :] = sample[0]
        batch_label[start:start + frame_num, :] = sample[1]
        start += frame_num
    return (batch_feature, batch_label)


def split_infer_result(infer_seq, lod):
    infer_batch = []
    for i in xrange(0, len(lod[0]) - 1):
        infer_batch.append(infer_seq[lod[0][i]:lod[0][i + 1]])
    return infer_batch


class DaemonProcessGroup(object):
    def __init__(self, proc_num, target, args):
        self._proc_num = proc_num
        self._workers = [
            Process(
                target=target, args=args) for _ in xrange(self._proc_num)
        ]

    def start_all(self):
        for w in self._workers:
            w.daemon = True
            w.start()

    @property
    def proc_num(self):
        return self._proc_num


class EpochEndSignal(object):
    pass


class CriticalException(Exception):
    pass


class SharedNDArray(object):
    """SharedNDArray utilizes shared memory to avoid data serialization when
    data object shared among different processes. We can reconstruct the
    `ndarray` when memory address, shape and dtype provided.

    Args:
        name (str): Address name of shared memory.
        whether_verify (bool): Whether to validate the writing operation.
    """

    def __init__(self, name, whether_verify=False):
        self._name = name
        self._shm = None
        self._buf = None
        self._array = np.zeros(1, dtype=np.float32)
        self._inited = False
        self._whether_verify = whether_verify

    def zeros_like(self, shape, dtype):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if self._inited:
            self._shm = posix_ipc.SharedMemory(self._name)
        else:
            self._shm = posix_ipc.SharedMemory(
                self._name, posix_ipc.O_CREAT, size=size)
        self._buf = mmap.mmap(self._shm.fd, size)
        self._array = np.ndarray(shape, dtype, self._buf, order='C')

    def copy(self, ndarray):
        size = int(np.prod(ndarray.shape)) * np.dtype(ndarray.dtype).itemsize
        self.zeros_like(ndarray.shape, ndarray.dtype)
        self._array[:] = ndarray
        self._buf.flush()
        self._inited = True

        if self._whether_verify:
            shm = posix_ipc.SharedMemory(self._name)
            buf = mmap.mmap(shm.fd, size)
            array = np.ndarray(ndarray.shape, ndarray.dtype, buf, order='C')
            np.testing.assert_array_equal(array, ndarray)

    @property
    def ndarray(self):
        return self._array

    def recycle(self, pool):
        self._buf.close()
        self._shm.close_fd()
        self._inited = False
        pool[self._name] = self

    def __getstate__(self):
        return (self._name, self._array.shape, self._array.dtype, self._inited,
                self._whether_verify)

    def __setstate__(self, state):
        self._name = state[0]
        self._inited = state[3]
        self.zeros_like(state[1], state[2])
        self._whether_verify = state[4]


class SharedMemoryPoolManager(object):
    """SharedMemoryPoolManager maintains a multiprocessing.Manager.dict object.
    All available addresses are allocated once and will be reused. Though this
    class is not process-safe, the pool can be shared between processes. All
    shared memory should be unlinked before the main process exited.

    Args:
        pool_size (int): Size of shared memory pool.
        manager (dict): A multiprocessing.Manager object, the pool is
                        maintained by the proxy process.
        name_prefix (str): Address prefix of shared memory.
    """

    def __init__(self, pool_size, manager, name_prefix='/deep_asr'):
        self._names = []
        self._dict = manager.dict()

        for i in xrange(pool_size):
            name = name_prefix + '_' + str(i)
            self._dict[name] = SharedNDArray(name)
            self._names.append(name)

    @property
    def pool(self):
        return self._dict

    def __del__(self):
        for name in self._names:
            # have to unlink the shared memory
            posix_ipc.unlink_shared_memory(name)


def suppress_signal(signo, stack_frame):
    pass


def suppress_complaints(verbose, notify=None):
    def decorator_maker(func):
        def suppress_warpper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except:
                et, ev, tb = sys.exc_info()

                if notify is not None:
                    notify(except_type=et, except_value=ev, traceback=tb)

                if verbose == 1 or isinstance(ev, CriticalException):
                    reraise(et, ev, Traceback(tb).as_traceback())

        return suppress_warpper

    return decorator_maker


class ForceExitWrapper(object):
    def __init__(self, exit_flag):
        self._exit_flag = exit_flag

    @suppress_complaints(verbose=0)
    def __call__(self, *args, **kwargs):
        self._exit_flag.value = True

    def __eq__(self, flag):
        return self._exit_flag.value == flag
