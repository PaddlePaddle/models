#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import UNK, TaskType, load_dic, sent2ids, logger


class Dataset(object):
    def __init__(self,
                 train_path,
                 test_path,
                 source_dic_path,
                 target_dic_path,
                 task_type=TaskType.RANK):
        self.train_path = train_path
        self.test_path = test_path
        self.source_dic_path = source_dic_path
        self.target_dic_path = target_dic_path
        self.task_type = task_type

        self.source_dic = load_dic(self.source_dic_path)
        self.target_dic = load_dic(self.target_dic_path)

        self.record_reader = self._read_classification_record \
                                if self.task_type == TaskType.CLASSFICATION \
                                        else self._read_rank_record

    def train(self):
        logger.info("[reader] load trainset from %s" % self.train_path)
        with open(self.train_path) as f:
            for line_id, line in enumerate(f):
                yield self.record_reader(line)

    def test(self):
        logger.info("[reader] load testset from %s" % self.test_path)
        with open(self.test_path) as f:
            for line_id, line in enumerate(f):
                yield self.record_reader(line)

    def _read_classification_record(self, line):
        '''
        data format:
            <source words> [TAB] <target words> [TAB] <label>

        @line: str
            a string line which represent a record.
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 3, "wrong format for classification\n" + \
            "the format shoud be " +\
            "<source words> [TAB] <target words> [TAB] <label>'"
        source = sent2ids(fs[0], self.source_dic)
        target = sent2ids(fs[1], self.target_dic)
        label = int(fs[2])
        return (source, target, label, )

    def _read_rank_record(self, line):
        '''
        data format:
            <source words> [TAB] <left_target words> [TAB] <right_target words> [TAB] <label>
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 4, "wrong format for rank\n" + \
            "the format should be " +\
            "<source words> [TAB] <left_target words> [TAB] <right_target words> [TAB] <label>"

        source = sent2ids(fs[0], self.source_dic)
        left_target = sent2ids(fs[1], self.target_dic)
        right_target = sent2ids(fs[2], self.target_dic)
        label = int(fs[3])

        return (source, left_target, right_target, label)


if __name__ == '__main__':
    path = './data/classification/train.txt'
    test_path = './data/classification/test.txt'
    source_dic = './data/vocab.txt'
    dataset = Dataset(path, test_path, source_dic, source_dic,
                      TaskType.CLASSFICATION)

    for rcd in dataset.train():
        print rcd
    # for i in range(10):
    #     print i, dataset.train().next()
