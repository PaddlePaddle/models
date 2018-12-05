import paddle
import time
import os
import paddle.fluid as fluid
import numpy as np
from Queue import PriorityQueue
import logging
import argparse
from sklearn.metrics.pairwise import cosine_similarity

word_to_id = dict()
id_to_word = dict()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec infer example")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/1-billion_dict',
        help="The path of training dataset")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help="The path for model to store (with infer_once please set specify dir to models) (default: models)"
    )
    parser.add_argument(
        '--rank_num',
        type=int,
        default=4,
        help="find rank_num-nearest result for test (default: 4)")
    parser.add_argument(
        '--infer_once',
        action='store_true',
        required=False,
        default=False,
        help='if using infer_once, (default: False)')
    parser.add_argument(
        '--infer_during_train',
        action='store_true',
        required=False,
        default=True,
        help='if using infer_during_train, (default: True)')

    return parser.parse_args()


def BuildWord_IdMap(dict_path):
    with open(dict_path + "_word_to_id_", 'r') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]


def inference_prog():
    fluid.layers.create_parameter(
        shape=[1, 1], dtype='float32', name="embeding")


def build_test_case(emb):
    emb1 = emb[word_to_id['boy']] - emb[word_to_id['girl']] + emb[word_to_id[
        'aunt']]
    desc1 = "boy - girl + aunt = uncle"
    emb2 = emb[word_to_id['brother']] - emb[word_to_id['sister']] + emb[
        word_to_id['sisters']]
    desc2 = "brother - sister + sisters = brothers"
    emb3 = emb[word_to_id['king']] - emb[word_to_id['queen']] + emb[word_to_id[
        'woman']]
    desc3 = "king - queen + woman = man"
    emb4 = emb[word_to_id['reluctant']] - emb[word_to_id['reluctantly']] + emb[
        word_to_id['slowly']]
    desc4 = "reluctant - reluctantly + slowly = slow"
    emb5 = emb[word_to_id['old']] - emb[word_to_id['older']] + emb[word_to_id[
        'deeper']]
    desc5 = "old - older + deeper = deep"
    return [[emb1, desc1], [emb2, desc2], [emb3, desc3], [emb4, desc4],
            [emb5, desc5]]


def inference_test(scope, model_dir, args):
    BuildWord_IdMap(args.dict_path)
    logger.info("model_dir is: {}".format(model_dir + "/"))
    emb = np.array(scope.find_var("embeding").get_tensor())
    test_cases = build_test_case(emb)
    logger.info("inference result: ====================")
    for case in test_cases:
        pq = topK(args.rank_num, emb, case[0])
        logger.info("Test result for {}".format(case[1]))
        pq_tmps = list()
        for i in range(args.rank_num):
            pq_tmps.append(pq.get())
        for i in range(len(pq_tmps)):
            logger.info("{} nearest is {}, rate is {}".format(i, id_to_word[
                pq_tmps[len(pq_tmps) - 1 - i].id], pq_tmps[len(pq_tmps) - 1 - i]
                                                              .priority))
        del pq_tmps[:]


class PQ_Entry(object):
    def __init__(self, cos_similarity, id):
        self.priority = cos_similarity
        self.id = id

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)


def topK(k, emb, test_emb):
    pq = PriorityQueue(k + 1)
    if len(emb) <= k:
        for i in range(len(emb)):
            x = cosine_similarity([emb[i]], [test_emb])
            pq.put(PQ_Entry(x, i))
        return pq

    for i in range(len(emb)):
        x = cosine_similarity([emb[i]], [test_emb])
        pq_e = PQ_Entry(x, i)
        if pq.full():
            pq.get()
        pq.put(pq_e)
    pq.get()
    return pq


def infer_during_train(args):
    model_file_list = list()
    exe = fluid.Executor(fluid.CPUPlace())
    Scope = fluid.Scope()
    inference_prog()
    solved_new = True
    while True:
        time.sleep(60)
        current_list = os.listdir(args.model_output_dir)
        # logger.info("current_list is : {}".format(current_list))
        # logger.info("model_file_list is : {}".format(model_file_list))
        if set(model_file_list) == set(current_list):
            if solved_new:
                solved_new = False
                logger.info("No New models created")
            pass
        else:
            solved_new = True
            increment_models = list()
            for f in current_list:
                if f not in model_file_list:
                    increment_models.append(f)
            logger.info("increment_models is : {}".format(increment_models))
            for model in increment_models:
                model_dir = args.model_output_dir + "/" + model
                if os.path.exists(model_dir + "/_success"):
                    logger.info("using models from " + model_dir)
                    with fluid.scope_guard(Scope):
                        fluid.io.load_persistables(
                            executor=exe, dirname=model_dir + "/")
                        inference_test(Scope, model_dir, args)
            model_file_list = current_list


def infer_once(args):
    # check models file has already been finished
    if os.path.exists(args.model_output_dir + "/_success"):
        logger.info("using models from " + args.model_output_dir)
        exe = fluid.Executor(fluid.CPUPlace())
        Scope = fluid.Scope()
        inference_prog()
        with fluid.scope_guard(Scope):
            fluid.io.load_persistables(
                executor=exe, dirname=args.model_output_dir + "/")
            inference_test(Scope, args.model_output_dir, args)


if __name__ == '__main__':
    args = parse_args()
    # while setting infer_once please specify the dir to models file with --model_output_dir
    if args.infer_once:
        infer_once(args)
    if args.infer_during_train:
        infer_during_train(args)
