import os
import math
import numpy as np

import paddle.v2 as paddle

from utils import logger, load_reverse_dict

__all__ = ["BeamSearch"]


class BeamSearch(object):
    """
    Generating sequence by beam search
    NOTE: this class only implements generating one sentence at a time.
    """

    def __init__(self, inferer, word_dict_file, beam_size=1, max_gen_len=100):
        """
        constructor method.

        :param inferer: object of paddle.Inference that represents the entire
            network to forward compute the test batch
        :type inferer: paddle.Inference
        :param word_dict_file: path of word dictionary file
        :type word_dict_file: str
        :param beam_size: expansion width in each iteration
        :type param beam_size: int
        :param max_gen_len: the maximum number of iterations
        :type max_gen_len: int
        """
        self.inferer = inferer
        self.beam_size = beam_size
        self.max_gen_len = max_gen_len
        self.ids_2_word = load_reverse_dict(word_dict_file)
        logger.info("dictionay len = %d" % (len(self.ids_2_word)))

        try:
            self.eos_id = next(x[0] for x in self.ids_2_word.iteritems()
                               if x[1] == "<e>")
            self.unk_id = next(x[0] for x in self.ids_2_word.iteritems()
                               if x[1] == "<unk>")
        except StopIteration:
            logger.fatal(("the word dictionay must contain an ending mark "
                          "in the text generation task."))

        self.candidate_paths = []
        self.final_paths = []

    def _top_k(self, softmax_out, k):
        """
        get indices of the words with k highest probablities.
        NOTE: <unk> will be excluded if it is among the top k words, then word
        with (k + 1)th highest probability will be returned.

        :param softmax_out: probablity over the dictionary
        :type softmax_out: narray
        :param k: number of word indices to return
        :type k: int
        :return: indices of k words with highest probablities.
        :rtype: list
        """
        ids = softmax_out.argsort()[::-1]
        return ids[ids != self.unk_id][:k]

    def _forward_batch(self, batch):
        """
        forward a test batch.

        :params batch: the input data batch
        :type batch: list
        :return: probablities of the predicted word
        :rtype: ndarray
        """
        return self.inferer.infer(input=batch, field=["value"])

    def _beam_expand(self, next_word_prob):
        """
        In every iteration step, the model predicts the possible next words.
        For each input sentence, the top k words is added to end of the original
        sentence to form a new generated sentence.

        :param next_word_prob: probablities of the next words
        :type next_word_prob: ndarray
        :return: the expanded new sentences.
        :rtype: list
        """
        assert len(next_word_prob) == len(self.candidate_paths), (
            "Wrong forward computing results!")
        top_beam_words = np.apply_along_axis(self._top_k, 1, next_word_prob,
                                             self.beam_size)
        new_paths = []
        for i, words in enumerate(top_beam_words):
            old_path = self.candidate_paths[i]
            for w in words:
                log_prob = old_path["log_prob"] + math.log(next_word_prob[i][w])
                gen_ids = old_path["ids"] + [w]
                if w == self.eos_id:
                    self.final_paths.append({
                        "log_prob": log_prob,
                        "ids": gen_ids
                    })
                else:
                    new_paths.append({"log_prob": log_prob, "ids": gen_ids})
        return new_paths

    def _beam_shrink(self, new_paths):
        """
        to return the top beam_size generated sequences with the highest
        probabilities at the end of evey generation iteration.

        :param new_paths: all possible generated sentences
        :type new_paths: list
        :return: a state flag to indicate whether to stop beam search
        :rtype: bool
        """

        if len(self.final_paths) >= self.beam_size:
            max_candidate_log_prob = max(
                new_paths, key=lambda x: x["log_prob"])["log_prob"]
            min_complete_path_log_prob = min(
                self.final_paths, key=lambda x: x["log_prob"])["log_prob"]
            if min_complete_path_log_prob >= max_candidate_log_prob:
                return True

        new_paths.sort(key=lambda x: x["log_prob"], reverse=True)
        self.candidate_paths = new_paths[:self.beam_size]
        return False

    def gen_a_sentence(self, input_sentence):
        """
        generating sequence for an given input

        :param input_sentence: one input_sentence
        :type input_sentence: list
        :return: the generated word sequences
        :rtype: list
        """
        self.candidate_paths = [{"log_prob": 0., "ids": input_sentence}]
        input_len = len(input_sentence)

        for i in range(self.max_gen_len):
            next_word_prob = self._forward_batch(
                [[x["ids"]] for x in self.candidate_paths])
            new_paths = self._beam_expand(next_word_prob)

            min_candidate_log_prob = min(
                new_paths, key=lambda x: x["log_prob"])["log_prob"]

            path_to_remove = [
                path for path in self.final_paths
                if path["log_prob"] < min_candidate_log_prob
            ]
            for p in path_to_remove:
                self.final_paths.remove(p)

            if self._beam_shrink(new_paths):
                self.candidate_paths = []
                break

        gen_ids = sorted(
            self.final_paths + self.candidate_paths,
            key=lambda x: x["log_prob"],
            reverse=True)[:self.beam_size]
        self.final_paths = []

        def _to_str(x):
            text = " ".join(self.ids_2_word[idx]
                            for idx in x["ids"][input_len:])
            return "%.4f\t%s" % (x["log_prob"], text)

        return map(_to_str, gen_ids)
