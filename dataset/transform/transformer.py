# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#   transform samples in 'source' using 'mapper'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid
import logging
from ..dataset import Dataset

logger = logging.getLogger(__name__)

class Transformer(Dataset):
    """ simple transformer without any workers to accelerate the processing
    """
    def __init__(self, source, mapper):
        self._source = source
        self._mapper = mapper

    def next(self):
        sample = self._source.next()
        return self._mapper(sample)

    def reset(self):
        self._source.reset()

    def drained(self):
        return self._source.drained()

    def epoch_id(self):
        return self._source.epoch_id()


class EndSignal(object):
    def __init__(self, errno=0, errmsg=''):
        self.errno = errno
        self.errmsg = errmsg


class FastTransformer(Dataset):
    """ a fast transformer using multiple workers (threads or processes),
        note that this class is not thread-safe
    """
    def __init__(self, source, mapper, worker_args):
        """ init
        """
        args = {'bufsize': 100, 'worker_num': 8}
        args.update(worker_args)
        self._worker_args = args
        self._started = False
        self._source = source
        self._mapper = mapper
        self._setup()

    def _setup(self):
        """ setup input/output queues and workers
        """
        try:
            from Queue import Queue
        except ImportError:
            from queue import Queue
        from threading import Thread
        from threading import Event

        bufsize = self._worker_args['bufsize']
        consumer_num = self._worker_args['worker_num']
        self._inq = Queue(bufsize)
        self._outq = Queue(bufsize)

        id = str(uuid.uuid4())[-3:]
        self._producer = Thread(
            target=self._produce,
            args=('producer-' + id, self._source, self._inq))
        self._producer.daemon = True

        self._consumers = []
        for i in range(consumer_num):
            p = Thread(
                target=self._consume,
                args=('consumer-' + id + '_%d' % (i),
                    self._inq, self._outq, self._mapper))
            self._consumers.append(p)
            p.daemon = True

        self._epoch = -1
        self._feeding_ev = Event()
        self._produced = 0 # produced sample in self._produce
        self._consumed = 0 # consumed sample in self.next
        self._stopped_consumers = 0

    def _produce(self, id, source, inq):
        """ fetch data from source and feed it to 'inq' queue
        """
        while True:
            self._feeding_ev.wait()
            try:
                inq.put(source.next())
                self._produced += 1
            except StopIteration as e:
                self._feeding_ev.clear()
                self._feeding_ev.wait() # wait other guy to wake up me
                logger.debug('producer[%s] begin another epoch to produce samples' % (id))
            except Exception as e:
                inq.put(EndSignal(-1,
                    'failed to produce with exception[%s] in producer[%s]' % (str(e), id)))
                break

        logger.debug('producer[%s] go to eixt' % (id))

    def _consume(self, id, inq, outq, mapper):
        """ fetch data from 'inq', then process it for result, 
            finally put result to 'outq'
        """
        while True:
            sample = inq.get()
            if isinstance(sample, EndSignal):
                sample.errmsg += '[consumer[%s] exit]' % (id)
                outq.put(sample)
                logger.debug('received end signal, so exit consumer[%s]' % (id))
                break

            try:
                result = mapper(sample)
                outq.put(result)
            except Exception as e:
                outq.put(EndSignal(-1, 'failed to do mapper with '
                    'exception[%s] in consumer[%s]' % (str(e), id)))
                break

    def drained(self):
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._source.drained() and \
            self._produced == self._consumed

    def next(self):
        """ get next transformed sample
        """
        if self._epoch < 0:
            self.reset()

        if self.drained():
            raise StopIteration()

        while True:
            sample = self._outq.get()
            if isinstance(sample, EndSignal):
                self._stopped_consumers += 1
                if sample.errno != 0:
                    logger.warn('error happened in consumer with errmsg[%s]' 
                        % (sample.errmsg))

                if self._stopped_consumers < len(self._consumers):
                    self._inq.put(sample)
                else:
                    raise ValueError('all consumers has exited, no more samples')
            else:
                self._consumed += 1
                return sample

    def reset(self):
        """ reset for a new epoch of samples
        """
        if self._epoch < 0:
            self._epoch = 0
            for p in self._consumers:
                p.start()
            self._producer.start()
        else:
            if not self.drained():
                logger.warn('reset before epoch[%d] has been finished!' % self._epoch)
                self._produced = self._produced - self._consumed
            else:
                self._produced = 0

            self._epoch += 1

        assert self._stopped_consumers == 0, 'some consumer has already exit,'\
            'cannot begin another epoch'

        self._source.reset()
        self._consumed = 0
        self._feeding_ev.set()

    def size(self):
        """ get number of samples
        """
        return self._source.size()

    def epoch_id(self):
        return self._source.epoch_id()
