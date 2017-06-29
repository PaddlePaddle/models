import logging

UNK = 0

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)


class TaskType:
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
        return TaskType(TaskType.TRAIN_MODE)

    @staticmethod
    def create_test():
        return TaskType(TaskType.TEST_MODE)

    @staticmethod
    def create_infer():
        return TaskType(TaskType.INFER_MODE)


class ModelType:
    CLASSIFICATION = 0
    RANK = 1

    def __init__(self, mode):
        self.mode = mode

    def is_classification(self):
        return self.mode == self.CLASSIFICATION

    def is_rank(self):
        return self.mode == self.RANK

    @staticmethod
    def create_classification():
        return ModelType(ModelType.CLASSIFICATION)

    @staticmethod
    def create_rank():
        return ModelType(ModelType.RANK)


def sent2ids(sent, vocab):
    '''
    transform a sentence to a list of ids.

    @sent: str
        a sentence.
    @vocab: dict
        a word dic
    '''
    return [vocab.get(w, UNK) for w in sent.split()]


def load_dic(path):
    '''
    word dic format:
      each line is a word
    '''
    dic = {}
    with open(path) as f:
        for id, line in enumerate(f):
            w = line.strip()
            dic[w] = id
    return dic
