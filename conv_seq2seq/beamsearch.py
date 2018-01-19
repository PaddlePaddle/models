#coding=utf-8

import sys
import time
import math
import numpy as np

import reader


class BeamSearch(object):
    """
    Generate sequence by beam search
    """

    def __init__(self,
                 inferer,
                 trg_dict,
                 pos_size,
                 padding_num,
                 batch_size=1,
                 beam_size=1,
                 max_len=100):
        self.inferer = inferer
        self.trg_dict = trg_dict
        self.reverse_trg_dict = reader.get_reverse_dict(trg_dict)
        self.word_padding = trg_dict.__len__()
        self.pos_size = pos_size
        self.pos_padding = pos_size
        self.padding_num = padding_num
        self.win_len = padding_num + 1
        self.max_len = max_len
        self.batch_size = batch_size
        self.beam_size = beam_size

    def get_beam_input(self, batch, sample_list):
        """
        Get input for generation at the current iteration.
        """
        beam_input = []

        for sample_id in sample_list:
            for path in self.candidate_path[sample_id]:
                if len(path['seq']) < self.win_len:
                    cur_trg = [self.word_padding] * (
                        self.win_len - len(path['seq']) - 1
                    ) + [self.trg_dict['<s>']] + path['seq']
                    cur_trg_pos = [self.pos_padding] * (
                        self.win_len - len(path['seq']) - 1) + [0] + range(
                            1, len(path['seq']) + 1)
                else:
                    cur_trg = path['seq'][-self.win_len:]
                    cur_trg_pos = range(
                        len(path['seq']) + 1 - self.win_len,
                        len(path['seq']) + 1)

                beam_input.append(batch[sample_id] + [cur_trg] + [cur_trg_pos])

        return beam_input

    def get_prob(self, beam_input):
        """
        Get the probabilities of all possible tokens.
        """
        row_list = [j * self.win_len for j in range(len(beam_input))]
        prob = self.inferer.infer(beam_input, field='value')[row_list, :]
        return prob

    def _top_k(self, prob, k):
        """
        Get indices of the words with k highest probablities.
        """
        return prob.argsort()[-k:][::-1]

    def beam_expand(self, prob, sample_list):
        """
        In every iteration step, the model predicts the possible next words.
        For each input sentence, the top beam_size words are selected as candidates.
        """
        top_words = np.apply_along_axis(self._top_k, 1, prob, self.beam_size)

        candidate_words = [[]] * len(self.candidate_path)
        idx = 0

        for sample_id in sample_list:
            for seq_id, path in enumerate(self.candidate_path[sample_id]):
                for w in top_words[idx, :]:
                    score = path['score'] + math.log(prob[idx, w])
                    candidate_words[sample_id] = candidate_words[sample_id] + [{
                        'word': w,
                        'score': score,
                        'seq_id': seq_id
                    }]
                idx = idx + 1

        return candidate_words

    def beam_shrink(self, candidate_words, sample_list):
        """
        Pruning process of the beam search. During the process, beam_size most post possible
        sequences are selected for the beam in the next generation.
        """
        new_path = [[]] * len(self.candidate_path)

        for sample_id in sample_list:
            beam_words = sorted(
                candidate_words[sample_id],
                key=lambda x: x['score'],
                reverse=True)[:self.beam_size]

            complete_seq_min_score = None
            complete_path_num = len(self.complete_path[sample_id])

            if complete_path_num > 0:
                complete_seq_min_score = min(self.complete_path[sample_id],
                                             key=lambda x: x['score'])['score']
                if complete_path_num >= self.beam_size:
                    beam_words_max_score = beam_words[0]['score']
                    if beam_words_max_score < complete_seq_min_score:
                        continue

            for w in beam_words:

                if w['word'] == self.trg_dict['<e>']:
                    if complete_path_num < self.beam_size or complete_seq_min_score <= w[
                            'score']:

                        seq = self.candidate_path[sample_id][w['seq_id']]['seq']
                        self.complete_path[sample_id] = self.complete_path[
                            sample_id] + [{
                                'seq': seq,
                                'score': w['score']
                            }]

                        if complete_seq_min_score is None or complete_seq_min_score > w[
                                'score']:
                            complete_seq_min_score = w['score']
                else:
                    seq = self.candidate_path[sample_id][w['seq_id']]['seq'] + [
                        w['word']
                    ]
                    new_path[sample_id] = new_path[sample_id] + [{
                        'seq': seq,
                        'score': w['score']
                    }]

        return new_path

    def search_one_batch(self, batch):
        """
        Perform beam search on one mini-batch.
        """
        real_size = len(batch)
        self.candidate_path = [[{'seq': [], 'score': 0.}]] * real_size
        self.complete_path = [[]] * real_size
        sample_list = range(real_size)

        for i in xrange(self.max_len):
            beam_input = self.get_beam_input(batch, sample_list)
            prob = self.get_prob(beam_input)

            candidate_words = self.beam_expand(prob, sample_list)
            new_path = self.beam_shrink(candidate_words, sample_list)
            self.candidate_path = new_path
            sample_list = [
                sample_id for sample_id in sample_list
                if len(new_path[sample_id]) > 0
            ]

            if len(sample_list) == 0:
                break

        final_path = []
        for i in xrange(real_size):
            top_path = sorted(
                self.complete_path[i] + self.candidate_path[i],
                key=lambda x: x['score'],
                reverse=True)[:self.beam_size]
            final_path.append(top_path)
        return final_path

    def search(self, infer_data):
        """
        Perform beam search on all data.
        """

        def _to_sentence(seq):
            raw_sentence = [self.reverse_trg_dict[id] for id in seq]
            sentence = " ".join(raw_sentence)
            return sentence

        for pos in xrange(0, len(infer_data), self.batch_size):
            batch = infer_data[pos:min(pos + self.batch_size, len(infer_data))]
            self.final_path = self.search_one_batch(batch)
            for top_path in self.final_path:
                print _to_sentence(top_path[0]['seq'])
            sys.stdout.flush()
