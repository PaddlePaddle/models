#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


class TaskMode(object):
    """
    TaskMode
    """
    TRAIN_MODE = 0
    TEST_MODE = 1
    INFER_MODE = 2

    def __init__(self, mode):
        """

        :param mode:
        """
        self.mode = mode

    def is_train(self):
        """

        :return:
        """
        return self.mode == self.TRAIN_MODE

    def is_test(self):
        """

        :return:
        """
        return self.mode == self.TEST_MODE

    def is_infer(self):
        """

        :return:
        """
        return self.mode == self.INFER_MODE

    @staticmethod
    def create_train():
        """

        :return:
        """
        return TaskMode(TaskMode.TRAIN_MODE)

    @staticmethod
    def create_test():
        """

        :return:
        """
        return TaskMode(TaskMode.TEST_MODE)

    @staticmethod
    def create_infer():
        """

        :return:
        """
        return TaskMode(TaskMode.INFER_MODE)
