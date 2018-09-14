import logging

logging.basicConfig()
logger = logging.getLogger("paddle")
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


class ModelType:
    CLASSIFICATION = 0
    REGRESSION = 1

    def __init__(self, mode):
        self.mode = mode

    def is_classification(self):
        return self.mode == self.CLASSIFICATION

    def is_regression(self):
        return self.mode == self.REGRESSION

    @staticmethod
    def create_classification():
        return ModelType(ModelType.CLASSIFICATION)

    @staticmethod
    def create_regression():
        return ModelType(ModelType.REGRESSION)


def load_dnn_input_record(sent):
    return map(int, sent.split())


def load_lr_input_record(sent):
    res = []
    for _ in [x.split(':') for x in sent.split()]:
        res.append((
            int(_[0]),
            float(_[1]), ))
    return res
