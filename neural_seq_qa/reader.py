import sys
import random
from itertools import izip
import json
import traceback

from datapoint import DataPoint, Evidence, EecommFeatures
import utils
from utils import logger

__all__ = [
    "Q_IDS", "E_IDS", "LABELS", "QE_COMM", "EE_COMM", "Q_IDS_STR", "E_IDS_STR",
    "LABELS_STR", "QE_COMM_STR", "EE_COMM_STR", "Settings", "create_reader"
]

# slot names
Q_IDS_STR = "q_ids"
E_IDS_STR = "e_ids"
LABELS_STR = "labels"
QE_COMM_STR = "qe.comm"
EE_COMM_STR = "ee.comm"

Q_IDS = 0
E_IDS = 1
LABELS = 2
QE_COMM = 3
EE_COMM = 4

NO_ANSWER = "no_answer"


class Settings(object):
    """
    class for storing settings
    """

    def __init__(self,
                 vocab,
                 is_training,
                 label_schema="BIO2",
                 negative_sample_ratio=0.2,
                 hit_ans_negative_sample_ratio=0.25,
                 keep_first_b=False,
                 seed=31425926):
        """
        Init function

        :param vocab: word dict
        :type vocab: dict
        :param is_training: True for training
        :type is_training: bool
        :param label_schema: label schema, valid values are BIO and BIO2,
            the default value is BIO2
        :type label_schema: str
        :param negative_sample_ratio: the ratio of negative samples used in
            training, the default value is 0.2
        :type negative_sample_ratio: float
        :param hit_ans_negative_sample_ratio: the ratio of negative samples 
            that contain golden answer string, the default value is 0.25
        :type hit_ans_negative_sample_ratio: float
        :param keep_first_b: only keep the first B in golden tag sequence,
            the default value is False
        :type keep_first_b: bool
        :param seed: random seed, the default value is 31425926
        :type seed: int
        """
        self.negative_sample_ratio = negative_sample_ratio
        self.hit_ans_negative_sample_ratio = hit_ans_negative_sample_ratio
        self.keep_first_b = keep_first_b
        self.is_training = is_training
        self.vocab = vocab

        # set up label schema
        if label_schema == "BIO":
            B, I, O1, O2 = 0, 1, 2, 2
        elif label_schema == "BIO2":
            B, I, O1, O2 = 0, 1, 2, 3
        else:
            raise ValueError("label_schema should be BIO/BIO2")
        self.B, self.I, self.O1, self.O2 = B, I, O1, O2
        self.label_map = {
            "B": B,
            "I": I,
            "O1": O1,
            "O2": O2,
            "b": B,
            "i": I,
            "o1": O1,
            "o2": O2
        }
        self.label_num = len(set((B, I, O1, O2)))

        # id for OOV
        self.oov_id = 0

        # set up random seed
        random.seed(seed)

        # booking message
        logger.info("negative_sample_ratio: %f", negative_sample_ratio)
        logger.info("hit_ans_negative_sample_ratio: %f",
                    hit_ans_negative_sample_ratio)
        logger.info("keep_first_b: %s", keep_first_b)
        logger.info("data reader random seed: %d", seed)


class SampleStream(object):
    def __init__(self, filename, settings):
        self.filename = filename
        self.settings = settings

    def __iter__(self):
        return self.load_and_filter_samples(self.filename)

    def load_and_filter_samples(self, filename):
        def remove_extra_b(labels):
            if labels.count(self.settings.B) <= 1: return

            i = 0
            # find the first B
            while i < len(labels) and labels[i] == self.settings.O1:
                i += 1
            i += 1  # skip B
            # skip the following Is
            while i < len(labels) and labels[i] == self.settings.I:
                i += 1
            # change all the other tags to O2
            while i < len(labels):
                labels[i] = self.settings.O2
                i += 1

        def filter_and_preprocess_evidences(evidences):
            for i, evi in enumerate(evidences):
                # convert golden labels to labels ids
                if Evidence.GOLDEN_LABELS in evi:
                    labels = [self.settings.label_map[l] \
                                for l in evi[Evidence.GOLDEN_LABELS]]
                else:
                    labels = [self.settings.O1] * len(evi[Evidence.E_TOKENS])

                # determine the current evidence is negative or not
                answer_list = evi[Evidence.GOLDEN_ANSWERS]
                is_negative = len(answer_list) == 1 \
                                and "".join(answer_list[0]).lower() == NO_ANSWER

                # drop positive evidences that do not contain golden answer
                # matches in training
                is_all_o1 = labels.count(self.settings.O1) == len(labels)
                if self.settings.is_training and is_all_o1 and not is_negative:
                    evidences[i] = None  # dropped
                    continue

                if self.settings.keep_first_b:
                    remove_extra_b(labels)
                evi[Evidence.GOLDEN_LABELS] = labels

        def get_eecom_feats_list(cur_sample_is_negative, eecom_feats_list,
                                 evidences):
            if not self.settings.is_training:
                return [item[EecommFeatures.EECOMM_FEATURES] \
                           for item in eecom_feats_list]

            positive_eecom_feats_list = []
            negative_eecom_feats_list = []

            for eecom_feats_, other_evi in izip(eecom_feats_list, evidences):
                if not other_evi: continue

                eecom_feats = eecom_feats_[EecommFeatures.EECOMM_FEATURES]
                if not eecom_feats: continue

                other_evi_type = eecom_feats_[EecommFeatures.OTHER_E_TYPE]
                if cur_sample_is_negative and \
                        other_evi_type != Evidence.POSITIVE:
                    continue

                if other_evi_type == Evidence.POSITIVE:
                    positive_eecom_feats_list.append(eecom_feats)
                else:
                    negative_eecom_feats_list.append(eecom_feats)

            eecom_feats_list = positive_eecom_feats_list
            if negative_eecom_feats_list:
                eecom_feats_list += [negative_eecom_feats_list]

            return eecom_feats_list

        def process_tokens(data, tok_key):
            ids = [self.settings.vocab.get(token, self.settings.oov_id) \
                        for token in data[tok_key]]
            return ids

        def process_evi(q_ids, evi, evidences):
            e_ids = process_tokens(evi, Evidence.E_TOKENS)

            labels = evi[Evidence.GOLDEN_LABELS]
            qe_comm = evi[Evidence.QECOMM_FEATURES]
            sample_type = evi[Evidence.TYPE]

            ret = [None] * 5
            ret[Q_IDS] = q_ids
            ret[E_IDS] = e_ids
            ret[LABELS] = labels
            ret[QE_COMM] = qe_comm

            eecom_feats_list = get_eecom_feats_list(
                sample_type != Evidence.POSITIVE,
                evi[Evidence.EECOMM_FEATURES_LIST], evidences)
            if not eecom_feats_list:
                return None
            else:
                ret[EE_COMM] = eecom_feats_list
                return ret

        with utils.DotBar(utils.open_file(filename)) as f_:
            for q_idx, line in enumerate(f_):
                # parse json line
                try:
                    data = json.loads(line)
                except Exception:
                    logger.fatal("ERROR LINE: %s", line.strip())
                    traceback.print_exc()
                    continue

                # convert question tokens to ids
                q_ids = process_tokens(data, DataPoint.Q_TOKENS)

                # process evidences
                evidences = data[DataPoint.EVIDENCES]
                filter_and_preprocess_evidences(evidences)
                for evi in evidences:
                    if not evi: continue
                    sample = process_evi(q_ids, evi, evidences)
                    if sample: yield q_idx, sample, evi[Evidence.TYPE]


class DataReader(object):
    def __iter__(self):
        return self

    def _next(self):
        raise NotImplemented()

    def next(self):
        data_point = self._next()
        return self.post_process_sample(data_point)

    def post_process_sample(self, sample):
        ret = list(sample)

        # choose eecom features randomly
        eecom_feats = random.choice(sample[EE_COMM])
        if not isinstance(eecom_feats[0], int):
            # the other evidence is a negative evidence
            eecom_feats = random.choice(eecom_feats)
        ret[EE_COMM] = eecom_feats

        return ret


class TrainingDataReader(DataReader):
    def __init__(self, sample_stream, negative_ratio, hit_ans_negative_ratio):
        super(TrainingDataReader, self).__init__()
        self.positive_data = []
        self.hit_ans_negative_data = []
        self.other_negative_data = []

        self.negative_ratio = negative_ratio
        self.hit_ans_negative_ratio = hit_ans_negative_ratio

        self.p_idx = 0
        self.hit_idx = 0
        self.other_idx = 0

        self.load_samples(sample_stream)

    def add_data(self, positive, hit_negative, other_negative):
        if not positive: return
        self.positive_data.extend(positive)
        for samples, target_list in \
                zip((hit_negative, other_negative),
                    (self.hit_ans_negative_data, self.other_negative_data)):
            if not samples: continue
            # `0" is an index, further refer to _next_negative_data()
            target_list.append([samples, 0])

    def load_samples(self, sample_stream):
        logger.info("loading data...")
        last_q_id, positive, hit_negative, other_negative = None, [], [], []
        for q_id, sample, type_ in sample_stream:
            if not last_q_id and q_id != last_q_id:
                self.add_data(positive, hit_negative, other_negative)
                positive, hit_negative, other_negative = [], [], []

            last_q_id = q_id
            if type_ == Evidence.POSITIVE:
                positive.append(sample)
            elif type_ == Evidence.HIT_ANS_NEGATIVE:
                hit_negative.append(sample)
            elif type_ == Evidence.OTHER_NEGATIVE:
                other_negative.append(sample)
            else:
                raise ValueError("wrong type: %s" % str(type_))
        self.add_data(positive, hit_negative, other_negative)

        # we are not sure whether the input data is shuffled or not
        # so we shuffle them
        random.shuffle(self.positive_data)
        random.shuffle(self.hit_ans_negative_data)
        random.shuffle(self.other_negative_data)

        # set thresholds
        if len(self.positive_data) == 0:
            logger.fatal("zero positive sample")
            raise ValueError("zero positive sample")

        zero_hit = len(self.hit_ans_negative_data) == 0
        zero_other = len(self.other_negative_data) == 0

        if zero_hit and zero_other:
            logger.fatal("zero negative sample")
            raise ValueError("zero negative sample")

        if zero_hit:
            logger.warning("zero hit_ans_negative sample")
            self.hit_ans_neg_threshold = 0
        else:
            self.hit_ans_neg_threshold = \
                self.negative_ratio * self.hit_ans_negative_ratio

        self.other_neg_threshold = self.negative_ratio
        if zero_other:
            logger.warning("zero other_negative sample")
            self.hit_ans_neg_threshold = self.negative_ratio
        logger.info("loaded")

    def next_positive_data(self):
        if self.p_idx >= len(self.positive_data):
            random.shuffle(self.positive_data)
            self.p_idx = 0

        self.p_idx += 1
        return self.positive_data[self.p_idx - 1]

    def _next_negative_data(self, idx, negative_data):
        if idx >= len(negative_data):
            random.shuffle(negative_data)
            idx = 0

        # a negative evidence is sampled in two steps: 
        # step 1: sample a question uniformly
        # step 2: sample a negative evidence corresponding to the question
        #         uniformly
        # bundle -> (sample, idx)
        bundle = negative_data[idx]
        if bundle[1] >= len(bundle[0]):
            random.shuffle(bundle[0])
            bundle[1] = 0
        bundle[1] += 1
        return idx + 1, bundle[0][bundle[1] - 1]

    def next_hit_ans_negative_data(self):
        self.hit_idx, data = self._next_negative_data(
            self.hit_idx, self.hit_ans_negative_data)
        return data

    def next_other_negative_data(self):
        self.other_idx, data = self._next_negative_data(
            self.other_idx, self.other_negative_data)
        return data

    def _next(self):
        rand = random.random()
        if rand <= self.hit_ans_neg_threshold:
            return self.next_hit_ans_negative_data()
        elif rand < self.other_neg_threshold:
            return self.next_other_negative_data()
        else:
            return self.next_positive_data()


class TestDataReader(DataReader):
    def __init__(self, sample_stream):
        super(TestDataReader, self).__init__()
        self.data_generator = iter(sample_stream)

    def _next(self):
        q_idx, sample, type_ = self.data_generator.next()
        return sample


def create_reader(filename, settings, samples_per_pass=sys.maxint):
    if settings.is_training:
        training_reader = TrainingDataReader(
            SampleStream(filename, settings), settings.negative_sample_ratio,
            settings.hit_ans_negative_sample_ratio)

        def wrapper():
            for i, data in izip(xrange(samples_per_pass), training_reader):
                yield data

        return wrapper
    else:

        def wrapper():
            sample_stream = SampleStream(filename, settings)
            return TestDataReader(sample_stream)

        return wrapper
