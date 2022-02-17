import os
from collections import OrderedDict

from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import time
import numpy as np

import sys
import signal


# handle terminate reader process, do not print stack frame
def _reader_quit(signum, frame):
    print("Reader process exit.")
    sys.exit()


def _term_group(sig_num, frame):
    print('pid {} terminated, terminate group '
          '{}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _reader_quit)
signal.signal(signal.SIGINT, _term_group)


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir,
                                              self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir,
                                                    [l.name for l in loaders])

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10, 'print_stats': None, 'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        if loader.training:
            self.actor.train()
        else:
            self.actor.eval()

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            data = self.to_variable(data)
            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                loss.backward()
                apply_collective_grads = getattr(self.actor.net,
                                                 "apply_collective_grads", None)
                if callable(apply_collective_grads):
                    apply_collective_grads()
                self.optimizer.minimize(loss)
                self.actor.net.clear_gradients()

            # update statistics
            batch_size = data['train_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            self._print_stats(i, loader, batch_size)

            if i % loader.__len__() == 0:
                self.save_checkpoint()
                self._stats_new_epoch()
                self._write_tensorboard()
                return

    def to_variable(self, data_dict):
        keys = data_dict.keys()
        for k in keys:
            if k != "dataset":
                data_dict[k] = dygraph.to_variable(
                    np.array(data_dict[k]).astype(np.float32))
        return data_dict

    def to_array(self, data_dict):
        keys = data_dict.keys()
        for k in keys:
            if k != "dataset":
                data_dict[k] = data_dict[k].numpy()
        return data_dict

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()
        print('{}th epoch train / eval done!'.format(self.epoch))

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[
                loader.name] is None:
            self.stats[loader.name] = OrderedDict(
                {name: AverageMeter()
                 for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i,
                                                loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or
                        name in self.settings.print_stats) and hasattr(val,
                                                                       'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print_str += '%s: %.5f  ,  ' % ("time", batch_size / batch_fps *
                                            self.settings.print_interval)
            if loader.training:
                print_str += '%s: %f  ,  ' % ("lr", self.optimizer.current_step_lr())
            print(print_str[:-5])

    def _stats_new_epoch(self):
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name,
                                               self.settings.script_name,
                                               self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
        print('{}/{}'.format(self.settings.module_name,
                             self.settings.script_name))
