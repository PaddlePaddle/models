#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
data preprocess for pathquery datasets
"""
import os
import sys
import time
import logging
import argparse
from kbc_data_preprocess import write_vocab
from kbc_data_preprocess import load_vocab
from kbc_data_preprocess import generate_mask_type
from collections import defaultdict, Counter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

inverted = lambda r: r[:2] == '**'
invert = lambda r: r[2:] if inverted(r) else '**' + r


class EvalDataset(object):
    def __init__(self, train_file, test_file):
        self.spo_train_fp = train_file
        self.spo_test_fp = test_file
        train_triples = self._load_spo_triples(self.spo_train_fp)
        test_triples = self._load_spo_triples(self.spo_test_fp)
        #logger.debug(">>train triples cnt:%d" % len(train_triples))
        #logger.debug(">>test triples cnt:%d" % len(test_triples))
        _train_cnt = len(train_triples)
        all_triples = train_triples
        all_triples.update(test_triples)
        self.full_graph = Graph(all_triples)
        logger.debug(self.full_graph)

    def _load_spo_triples(self, spo_path):
        """
        :param spo_path:
        :return:  set of (s,r,t) original triples
        """
        logger.debug(">> Begin load base spo for %s at %s" %
                     (spo_path, time.ctime()))
        triples = set()
        for line in open(spo_path):
            segs = line.strip().split("\t")
            assert len(segs) == 3
            s, p, o = segs
            triples.add((s, p, o))
        logger.debug(">> Loaded spo triples :%s  cnt:%d" %
                     (spo_path, len(triples)))
        logger.debug(">> End load spo for %s at %s" % (spo_path, time.ctime()))
        return triples


class Graph(object):
    def __init__(self, triples):
        self.triples = triples
        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        logger.info(">> Begin building graph at %s" % (time.ctime()))
        self._node_set = set()
        for s, r, t in triples:
            relation_args[r]['s'].add(s)
            relation_args[r]['t'].add(t)
            neighbors[s][r].add(t)
            neighbors[t][invert(r)].add(s)
            self._node_set.add(t)
            self._node_set.add(s)

        def freeze(d):
            frozen = {}
            for key, subdict in d.iteritems():
                frozen[key] = {}
                for subkey, set_val in subdict.iteritems():
                    frozen[key][subkey] = tuple(set_val)
            return frozen

        self.neighbors = freeze(neighbors)
        self.relation_args = freeze(relation_args)
        logger.info(">> Done building graph at %s" % (time.ctime()))

    def __repr__(self):
        s = ""
        s += "graph.relations_args cnt %d\t" % len(self.relation_args)
        s += "graph.neighbors cnt %d\t" % len(self.neighbors)
        s += "graph.neighbors node set cnt %d" % len(self._node_set)
        return s

    def walk_all(self, start, path):
        """
        walk from start and get all the paths
        :param start:  start entity
        :param path: (r1, r2, ...,rk)
        :return: entities set for candidates path
        """
        set_s = set()
        set_t = set()
        set_s.add(start)
        for _, r in enumerate(path):
            if len(set_s) == 0:
                return set()
            for _s in set_s:
                if _s in self.neighbors and r in self.neighbors[_s]:
                    _tset = set(self.neighbors[_s][r])  #tupe to set
                    set_t.update(_tset)
            set_s = set_t.copy()
            set_t.clear()
        return set_s

    def repr_walk_all_ret(self, start, path, MAX_T=20):
        cand_set = self.walk_all(start, path)
        if len(cand_set) == 0:
            return ">>start{} path:{} end: EMPTY!".format(
                start, "->".join(list(path)))
        _len = len(cand_set) if len(cand_set) < MAX_T else MAX_T
        cand_node_str = ", ".join(cand_set[:_len])
        return ">>start{} path:{} end: {}".format(
            start, "->".join(list(path)), cand_node_str)

    def type_matching_entities(self, path, position="t"):
        assert (position == "t")
        if position == "t":
            r = path[-1]
        elif position == "s":
            r = path[0]
        else:
            logger.error(">>UNKNOWN position at type_matching_entities")
            raise ValueError(position)
        try:
            if not inverted(r):
                return r, self.relation_args[r][position]
            else:
                inv_pos = 's' if position == "t" else "t"
                return r, self.relation_args[invert(r)][inv_pos]
        except KeyError:
            logger.error(
                ">>UNKNOWN path value at type_matching_entities :%s from path:%s"
                % (r, path))
            return None, tuple()

    def is_trival_query(self, start, path):
        """
        :param path:
        :return: Boolean if True/False, is all candidates are right answers, return True
        """
        #todo: check right again
        cand_set = self.type_matching_entities(path, "t")
        ans_set = self.walk_all(start, path)
        _set = cand_set - ans_set
        if len(_set) == 0:
            return True
        else:
            return False


def get_unique_entities_relations(train_file, dev_file, test_file):
    entity_lst = dict()
    relation_lst = dict()
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        with open(input_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                entity_lst[tokens[0]] = len(entity_lst)
                entity_lst[tokens[2]] = len(entity_lst)
                relations = tokens[1].split(",")
                for relation in relations:
                    relation_lst[relation] = len(relation_lst)
    print(">> Number of unique entities: %s" % len(entity_lst))
    print(">> Number of unique relations: %s" % len(relation_lst))
    return entity_lst, relation_lst


def filter_base_data(raw_train_file, raw_dev_file, raw_test_file,
                     train_base_file, dev_base_file, test_base_file):
    def fil_base(input_file, output_file):
        fout = open(output_file, "w")
        base_n = 0
        with open(input_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                relations = tokens[1].split(",")
                if len(relations) == 1:
                    fout.write(line)
                    base_n += 1
        fout.close()
        return base_n

    train_base_n = fil_base(raw_train_file, train_base_file)
    dev_base_n = fil_base(raw_dev_file, dev_base_file)
    test_base_n = fil_base(raw_test_file, test_base_file)
    print(">> Train base cnt:%d" % train_base_n)
    print(">> Valid base cnt:%d" % dev_base_n)
    print(">> Test base cnt:%d" % test_base_n)


def generate_onlytail_mask_type(input_file, output_file):
    with open(output_file, "w") as fw:
        with open(input_file, "r") as fr:
            for line in fr.readlines():
                fw.write(line.strip('\r \n') + "\tMASK_TAIL\n")


def generate_eval_files(vocab_path, raw_test_file, train_base_file,
                        dev_base_file, test_base_file, sen_candli_file,
                        trivial_sen_file):
    token2id = load_vocab(vocab_path)

    eval_data = EvalDataset(train_base_file, test_base_file)

    fout_sen_cand = open(sen_candli_file, "w")
    fout_q_trival = open(trivial_sen_file, "w")

    sen_candli_cnt = trivial_sen_cnt = 0
    j = 0
    for line in open(raw_test_file):
        line = line.strip()
        j += 1
        segs = line.split("\t")
        s = segs[0]
        t = segs[2]
        path = tuple(segs[1].split(","))

        q_set = eval_data.full_graph.walk_all(s, path)
        r, cand_set = eval_data.full_graph.type_matching_entities(path, "t")
        cand_set = set(cand_set)
        neg_set = cand_set - q_set

        sen_tokens = []
        sen_tokens.append(line.split("\t")[0])
        sen_tokens.extend(line.split("\t")[1].split(","))
        sen_tokens.append(line.split("\t")[2])
        sen_id = [str(token2id[x]) for x in sen_tokens]
        if len(neg_set) == 0:
            trivial_sen_cnt += 1
            #fout_q_trival.write(line + "\n")
            fout_q_trival.write(" ".join(sen_id) + "\n")
        else:
            sen_candli_cnt += 1
            candli_id_set = [str(token2id[x]) for x in neg_set]
            sen_canli_str = "%s\t%s" % (" ".join(sen_id),
                                        " ".join(list(candli_id_set)))
            fout_sen_cand.write(sen_canli_str + "\n")

        if len(cand_set) < len(q_set):
            logger.error("ERROR! cand_set %d < q_set %d  at line[%d]:%s" %
                         (len(cand_set), len(q_set), j, line))
        if j % 100 == 0:
            logger.debug(" ...processing %d at %s" % (j, time.ctime()))
        if -100 > 0 and j >= 100:
            break
    logger.info(">> sen_canli_set count:%d " % sen_candli_cnt)
    logger.info(">> trivial sen count:%d " % trivial_sen_cnt)
    logger.info(">> Finish generate evaluation candidates for %s file at %s" %
                (raw_test_file, time.ctime()))


def pathquery_data_preprocess(raw_train_file, raw_dev_file, raw_test_file,
                              vocab_path, sen_candli_file, trivial_sen_file,
                              new_train_file, new_dev_file, new_test_file,
                              train_base_file, dev_base_file, test_base_file):
    entity_lst, relation_lst = get_unique_entities_relations(
        raw_train_file, raw_dev_file, raw_test_file)
    write_vocab(vocab_path, entity_lst, relation_lst)
    filter_base_data(raw_train_file, raw_dev_file, raw_test_file,
                     train_base_file, dev_base_file, test_base_file)
    generate_mask_type(raw_train_file, new_train_file)
    generate_onlytail_mask_type(raw_dev_file, new_dev_file)
    generate_onlytail_mask_type(raw_test_file, new_test_file)
    vocab = load_vocab(vocab_path)
    generate_eval_files(vocab_path, raw_test_file, train_base_file,
                        dev_base_file, test_base_file, sen_candli_file,
                        trivial_sen_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default=None,
        help="task name: fb15k, fb15k237, wn18rr, wn18, pathqueryFB, pathqueryWN"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        default=None,
        help="task data directory")
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="train",
        help="train file name, default train.txt")
    parser.add_argument(
        "--valid",
        type=str,
        required=False,
        default="dev",
        help="valid file name, default valid.txt")
    parser.add_argument(
        "--test",
        type=str,
        required=False,
        default="test",
        help="test file name, default test.txt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    task = args.task.lower()
    assert task in ["pathqueryfb", "pathquerywn"]
    raw_train_file = os.path.join(args.dir, args.train)
    raw_dev_file = os.path.join(args.dir, args.valid)
    raw_test_file = os.path.join(args.dir, args.test)

    new_train_file = os.path.join(args.dir, "train.coke.txt")
    new_test_file = os.path.join(args.dir, "test.coke.txt")
    new_dev_file = os.path.join(args.dir, "dev.coke.txt")

    vocab_file = os.path.join(args.dir, "vocab.txt")
    sen_candli_file = os.path.join(args.dir, "sen_candli.txt")
    trivial_sen_file = os.path.join(args.dir, "trivial_sen.txt")

    train_base_file = os.path.join(args.dir, "train.base.txt")
    test_base_file = os.path.join(args.dir, "test.base.txt")
    dev_base_file = os.path.join(args.dir, "dev.base.txt")

    pathquery_data_preprocess(raw_train_file, raw_dev_file, raw_test_file,
                              vocab_file, sen_candli_file, trivial_sen_file,
                              new_train_file, new_dev_file, new_test_file,
                              train_base_file, dev_base_file, test_base_file)
