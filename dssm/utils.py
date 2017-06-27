import logging

UNK = 0

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)


class TaskType:
    '''
    type of DSSM's task.
    '''
    # pairwise rank.
    RANK = 0
    # classification.
    CLASSFICATION = 1


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
