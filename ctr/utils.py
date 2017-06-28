import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)


class TaskMode:
    TRAIN_MODE = 0
    TEST_MODE = 1
    INFER_MODE = 2

    def __init__(self, mode):
        self.mode = mode

    def is_train(self):
        return self.mode == self.TRAIN_MODE

    def is_test(self):
        return self.mode == self.TEST_MODE

    def is_infer(self):
        return self.mode == self.INFER_MODE

    @staticmethod
    def create_train():
        return TaskMode(TaskMode.TRAIN_MODE)

    @staticmethod
    def create_test():
        return TaskMode(TaskMode.TEST_MODE)

    @staticmethod
    def create_infer():
        return TaskMode(TaskMode.INFER_MODE)
