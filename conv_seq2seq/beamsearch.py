#coding=utf-8

import sys
import time
import numpy as np


class BeamSearch(object):
    """
    Generate sequence by beam search
    NOTE: this class only implements generating one sentence at a time.
    """

    def __init__(self,
                 inferer,
                 trg_dict,
                 pos_size,
                 padding_num,
                 beam_size=1,
                 max_len=100):
        self.inferer = inferer
        self.trg_dict = trg_dict
        self.word_padding = trg_dict.__len__()
        self.pos_size = pos_size
        self.pos_padding = pos_size
        self.padding_num = padding_num
        self.win_len = padding_num + 1
        self.max_len = max_len
        self.beam_size = beam_size

    def get_beam_input(self, pre_beam_list, infer_data):
        """
        Get input for generation at the current iteration.
        """
        beam_input = []

        if len(pre_beam_list) == 0:
            cur_trg = [self.word_padding
                       ] * self.padding_num + [self.trg_dict['<s>']]
            cur_trg_pos = [self.pos_padding] * self.padding_num + [0]
            beam_input.append(infer_data + [cur_trg] + [cur_trg_pos])
        else:
            for seq in pre_beam_list:
                if len(seq) < self.win_len:
                    cur_trg = [self.word_padding] * (
                        self.win_len - len(seq) - 1
                    ) + [self.trg_dict['<s>']] + seq
                    cur_trg_pos = [self.pos_padding] * (
                        self.win_len - len(seq) - 1) + [0] + range(1,
                                                                   len(seq) + 1)
                else:
                    cur_trg = seq[-self.win_len:]
                    cur_trg_pos = range(
                        len(seq) + 1 - self.win_len, len(seq) + 1)

                beam_input.append(infer_data + [cur_trg] + [cur_trg_pos])
        return beam_input

    def get_prob(self, beam_input):
        """
        Get the probabilities of all possible tokens.
        """
        row_list = [j * self.win_len for j in range(len(beam_input))]
        prob = self.inferer.infer(beam_input, field='value')[row_list, :]
        return prob

    def get_candidate(self, pre_beam_list, pre_beam_score, prob):
        """
        Get top beam_size tokens and their scores for each beam.
        """
        if prob.ndim == 1:
            candidate_id = prob.argsort()[-self.beam_size:][::-1]
            candidate_log_prob = np.log(prob[candidate_id])
        else:
            candidate_id = prob.argsort()[:, -self.beam_size:][:, ::-1]
            candidate_log_prob = np.zeros_like(candidate_id).astype('float32')
            for j in range(len(pre_beam_list)):
                candidate_log_prob[j, :] = np.log(prob[j, candidate_id[j, :]])

        if pre_beam_score.size > 0:
            candidate_score = candidate_log_prob + pre_beam_score.reshape(
                (pre_beam_score.size, 1))
        else:
            candidate_score = candidate_log_prob

        return candidate_id, candidate_score

    def prune(self, candidate_id, candidate_score, pre_beam_list,
              completed_seq_list, completed_seq_score, completed_seq_min_score):
        """
        Pruning process of the beam search. During the process, beam_size most possible sequences
        are selected for the beam in the next iteration. Besides, their scores and the minimum score
        of the completed sequences are updated.
        """
        candidate_id = candidate_id.flatten()
        candidate_score = candidate_score.flatten()

        topk_idx = candidate_score.argsort()[-self.beam_size:][::-1].tolist()
        topk_seq_idx = [idx / self.beam_size for idx in topk_idx]

        next_beam = []
        beam_score = []
        for j in range(len(topk_idx)):
            if candidate_id[topk_idx[j]] == self.trg_dict['<e>']:
                if len(
                        completed_seq_list
                ) < self.beam_size or completed_seq_min_score <= candidate_score[
                        topk_idx[j]]:
                    completed_seq_list.append(pre_beam_list[topk_seq_idx[j]])
                    completed_seq_score.append(candidate_score[topk_idx[j]])

                    if completed_seq_min_score is None or (
                            completed_seq_min_score >=
                            candidate_score[topk_idx[j]] and
                            len(completed_seq_list) < self.beam_size):
                        completed_seq_min_score = candidate_score[topk_idx[j]]
            else:
                seq = pre_beam_list[topk_seq_idx[
                    j]] + [candidate_id[topk_idx[j]]]
                score = candidate_score[topk_idx[j]]
                next_beam.append(seq)
                beam_score.append(score)

        beam_score = np.array(beam_score)
        return next_beam, beam_score, completed_seq_min_score

    def search_one_sample(self, infer_data):
        """
        Beam search process for one sample.
        """
        completed_seq_list = []
        completed_seq_score = []
        completed_seq_min_score = None
        uncompleted_seq_list = [[]]
        uncompleted_seq_score = np.zeros(0)

        for i in xrange(self.max_len):
            beam_input = self.get_beam_input(uncompleted_seq_list, infer_data)

            prob = self.get_prob(beam_input)

            candidate_id, candidate_score = self.get_candidate(
                uncompleted_seq_list, uncompleted_seq_score, prob)

            uncompleted_seq_list, uncompleted_seq_score, completed_seq_min_score = self.prune(
                candidate_id, candidate_score, uncompleted_seq_list,
                completed_seq_list, completed_seq_score,
                completed_seq_min_score)

            if len(uncompleted_seq_list) == 0:
                break
            if len(completed_seq_list) >= self.beam_size:
                seq_max_score = uncompleted_seq_score.max()
                if seq_max_score < completed_seq_min_score:
                    uncompleted_seq_list = []
                    break

        final_seq_list = completed_seq_list + uncompleted_seq_list
        final_score = np.concatenate(
            (np.array(completed_seq_score), uncompleted_seq_score))
        max_id = final_score.argmax()
        top_seq = final_seq_list[max_id]
        return top_seq
