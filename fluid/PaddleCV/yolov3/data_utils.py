"""
This code is based on https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
"""

import time
import numpy as np
import threading
import multiprocessing
import cv2
try:
    import queue
except ImportError:
    import Queue as queue

import image_utils


class GeneratorEnqueuer(object):
    """
    Builds a queue out of a data generator.

    Args:
        generator: a generator function which endlessly yields data
        use_multiprocessing (bool): use multiprocessing if True,
            otherwise use threading.
        wait_time (float): time to sleep in-between calls to `put()`.
        random_seed (int): Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self,
                 generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queues = []
        self._manager = None
        self.seed = random_seed

    def start(self, workers=1, max_queue_size=10, random_sizes=[608]):
        """
        Start worker threads which add data from the generator into the queue.

        Args:
            workers (int): number of worker threads
            max_queue_size (int): queue size
                (when full, threads could block on `put()`)
        """

        self.random_sizes = random_sizes
        self.size_num = len(random_sizes)

        def data_generator_task():
            """
            Data generator task.
            """

            def task():
                if len(self.queues) > 0:
                    generator_output = next(self._generator)
                    queue_idx = 0
                    while(True):
                        if self.queues[queue_idx].full():
                            queue_idx = (queue_idx + 1) % self.size_num
                            continue
                        else:
                            size = self.random_sizes[queue_idx]
                            for g in generator_output:
                                g[0] = g[0].transpose((1, 2, 0))
                                g[0] = image_utils.random_interp(g[0], size, cv2.INTER_LINEAR)
                                g[0] = g[0].transpose((2, 0, 1))
                            try:
                                self.queues[queue_idx].put_nowait(generator_output)
                            except:
                                continue
                            else:
                                break


                    # self.queue.put((generator_output))
                else:
                    time.sleep(self.wait_time)

            if not self._use_multiprocessing:
                while not self._stop_event.is_set():
                    with self.genlock:
                        try:
                            task()
                        except Exception:
                            self._stop_event.set()
                            break
            else:
                while not self._stop_event.is_set():
                    try:
                        task()
                    except Exception:
                        self._stop_event.set()
                        break

        try:
            if self._use_multiprocessing:
                self._manager = multiprocessing.Manager()
                for i in range(self.size_num):
                    self.queues.append(self._manager.Queue(maxsize=max_queue_size))
                self._stop_event = multiprocessing.Event()
            else:
                self.genlock = threading.Lock()
                # self.queue = queue.Queue()
                for i in range(self.size_num):
                    self.queues.append(queue.Queue())
                self._stop_event = threading.Event()
            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.seed is not None:
                        self.seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        """
        Returns:
            bool: Whether the worker theads are running.
        """
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """
        Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.

        Args:
            timeout(int|None): maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()
        for thread in self._threads:
            if self._use_multiprocessing:
                if thread.is_alive():
                    thread.terminate()
            else:
                thread.join(timeout)
        if self._manager:
            self._manager.shutdown()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self, queue_idx):
        """
        Creates a generator to extract data from the queue.
        Skip the data if it is `None`.

        # Yields
            tuple of data in the queue.
        """
        while self.is_running():
            if not self.queues[queue_idx].empty():
                inputs = self.queues[queue_idx].get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
