import time
import os
import paddle.fluid as fluid
import numpy as np
from Queue import PriorityQueue
import logging
import argparse
import preprocess
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
    parser.add_argument(
        '--test_acc',
        action='store_true',
        required=False,
        default=True,
        help='if using test_files , (default: True)')
    parser.add_argument(
        '--test_files_dir',
        type=str,
        default='test',
        help="The path for test_files) (default: test)")
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=1000,
        help="test used batch size (default: 1000)")

    return parser.parse_args()


def BuildWord_IdMap(dict_path):
    with open(dict_path + "_word_to_id_", 'r') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]


def inference_prog():  # just to create program for test
    fluid.layers.create_parameter(
        shape=[1, 1], dtype='float32', name="embeding")


def build_test_case_from_file(args, emb):
    logger.info("test files dir: {}".format(args.test_files_dir))
    current_list = os.listdir(args.test_files_dir)
    logger.info("test files list: {}".format(current_list))
    test_cases = list()
    test_labels = list()
    exclude_lists = list()
    for file_dir in current_list:
        with open(args.test_files_dir + "/" + file_dir, 'r') as f:
            count = 0
            for line in f:
                if count == 0:
                    pass
                elif ':' in line:
                    logger.info("{}".format(line))
                    pass
                else:
                    line = preprocess.strip_lines(line, word_to_id)
                    test_case = emb[word_to_id[line.split()[0]]] - emb[
                        word_to_id[line.split()[1]]] + emb[word_to_id[
                            line.split()[2]]]
                    test_case_desc = line.split()[0] + " - " + line.split()[
                        1] + " + " + line.split()[2] + " = " + line.split()[3]
                    test_cases.append([test_case, test_case_desc])
                    test_labels.append(word_to_id[line.split()[3]])
                    exclude_lists.append([
                        word_to_id[line.split()[0]],
                        word_to_id[line.split()[1]], word_to_id[line.split()[2]]
                    ])
                count += 1
    return test_cases, test_labels, exclude_lists


def build_small_test_case(emb):
    emb1 = emb[word_to_id['boy']] - emb[word_to_id['girl']] + emb[word_to_id[
        'aunt']]
    desc1 = "boy - girl + aunt = uncle"
    label1 = word_to_id["uncle"]
    emb2 = emb[word_to_id['brother']] - emb[word_to_id['sister']] + emb[
        word_to_id['sisters']]
    desc2 = "brother - sister + sisters = brothers"
    label2 = word_to_id["brothers"]
    emb3 = emb[word_to_id['king']] - emb[word_to_id['queen']] + emb[word_to_id[
        'woman']]
    desc3 = "king - queen + woman = man"
    label3 = word_to_id["man"]
    emb4 = emb[word_to_id['reluctant']] - emb[word_to_id['reluctantly']] + emb[
        word_to_id['slowly']]
    desc4 = "reluctant - reluctantly + slowly = slow"
    label4 = word_to_id["slow"]
    emb5 = emb[word_to_id['old']] - emb[word_to_id['older']] + emb[word_to_id[
        'deeper']]
    desc5 = "old - older + deeper = deep"
    label5 = word_to_id["deep"]
    return [[emb1, desc1], [emb2, desc2], [emb3, desc3], [emb4, desc4],
            [emb5, desc5]], [label1, label2, label3, label4, label5]


def build_test_case(args, emb):
    if args.test_acc:
        return build_test_case_from_file(args, emb)
    else:
        return build_small_test_case(emb)


def inference_test(scope, model_dir, args):
    BuildWord_IdMap(args.dict_path)
    logger.info("model_dir is: {}".format(model_dir + "/"))
    emb = np.array(scope.find_var("embeding").get_tensor())
    logger.info("inference result: ====================")
    test_cases = list()
    test_labels = list()
    exclude_lists = list()
    if args.test_acc:
        test_cases, test_labels, exclude_lists = build_test_case(args, emb)
    else:
        test_cases, test_labels = build_test_case(args, emb)
        exclude_lists = [[-1]]
    accual_rank = 1 if args.test_acc else args.rank_num
    correct_num = 0
    for i in range(len(test_labels)):
        pq = None
        if args.test_acc:
            pq = topK(
                accual_rank,
                emb,
                test_cases[i][0],
                exclude_lists[i],
                is_acc=True)
        else:
            pq = pq = topK(
                accual_rank,
                emb,
                test_cases[i][0],
                exclude_lists[0],
                is_acc=False)
        logger.info("Test result for {}".format(test_cases[i][1]))
        for j in range(accual_rank):
            pq_tmps = pq.get()
            if (j == accual_rank - 1) and (
                    pq_tmps.id == test_labels[i]
            ):  # if the nearest word is what we want 
                correct_num += 1
            logger.info("{} nearest is {}, rate is {}".format(
                accual_rank - j, id_to_word[pq_tmps.id], pq_tmps.priority))
    acc = correct_num / len(test_labels)
    logger.info("Test acc is: {}, there are {} / {}}".format(acc, correct_num,
                                                             len(test_labels)))


class PQ_Entry(object):
    def __init__(self, cos_similarity, id):
        self.priority = cos_similarity
        self.id = id

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)


def topK(k, emb, test_emb, exclude_list, is_acc=False):
    pq = PriorityQueue(k + 1)
    while not pq.empty():
        try:
            pq.get(False)
        except Empty:
            continue
        pq.task_done()

    if len(emb) <= k:
        for i in range(len(emb)):
            x = cosine_similarity([emb[i]], [test_emb])
            pq.put(PQ_Entry(x, i))
        return pq

    for i in range(len(emb)):
        if is_acc and (i in exclude_list):
            pass
        else:
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
    elif args.infer_during_train:
        infer_during_train(args)
    else:
        pass
