#!/usr/bin/env python
#coding=utf-8
import pdb
import numpy as np

__all__ = ["BeamDecoding"]


class BeamDecoding(object):
    def __init__(self, documents, sentence_scores, selected_sentences,
                 start_scores, selected_starts, end_scores, selected_ends):
        self.documents = documents

        self.sentence_scores = sentence_scores
        self.selected_sentences = selected_sentences

        self.start_scores = start_scores
        self.selected_starts = selected_starts

        self.end_scores = end_scores
        self.selected_ends = selected_ends

        # sequence start position information for the three step search
        # beam1 is to search the sequence index
        self.beam1_seq_start_positions = []
        # beam2 is to search the start answer span
        self.beam2_seq_start_positions = []
        # beam3 is to search the end answer span
        self.beam3_seq_start_positions = []

        self.ans_per_sample_in_a_batch = [0]
        self.all_searched_ans = []

        self.final_ans = [[] for i in range(len(documents))]

    def _build_beam1_seq_info(self):
        self.beam1_seq_start_positions.append([0])
        for idx, one_doc in enumerate(self.documents):
            for sentence in one_doc:
                self.beam1_seq_start_positions[-1].append(
                    self.beam1_seq_start_positions[-1][-1] + len(sentence))

            if len(self.beam1_seq_start_positions) != len(self.documents):
                self.beam1_seq_start_positions.append(
                    [self.beam1_seq_start_positions[-1][-1]])

    def _build_beam2_seq_info(self):
        seq_num, beam_size = self.selected_sentences.shape
        self.beam2_seq_start_positions.append([0])
        for i in range(seq_num):
            for j in range(beam_size):
                selected_id = int(self.selected_sentences[i][j])
                if selected_id == -1: break
                seq_len = self.beam1_seq_start_positions[i][
                    selected_id + 1] - self.beam1_seq_start_positions[i][
                        selected_id]
                self.beam2_seq_start_positions[-1].append(
                    self.beam2_seq_start_positions[-1][-1] + seq_len)

            if len(self.beam2_seq_start_positions) != seq_num:
                self.beam2_seq_start_positions.append(
                    [self.beam2_seq_start_positions[-1][-1]])

    def _build_beam3_seq_info(self):
        seq_num_in_a_batch = len(self.documents)

        seq_id = 0
        sub_seq_id = 0
        sub_seq_count = len(self.beam2_seq_start_positions[seq_id]) - 1

        self.beam3_seq_start_positions.append([0])
        sub_seq_num, beam_size = self.selected_starts.shape
        for i in range(sub_seq_num):
            seq_len = self.beam2_seq_start_positions[seq_id][
                sub_seq_id + 1] - self.beam2_seq_start_positions[seq_id][
                    sub_seq_id]
            for j in range(beam_size):
                start_id = int(self.selected_starts[i][j])
                if start_id == -1: break

                self.beam3_seq_start_positions[-1].append(
                    self.beam3_seq_start_positions[-1][-1] + seq_len - start_id)

            sub_seq_id += 1
            if sub_seq_id == sub_seq_count:
                if len(self.beam3_seq_start_positions) != seq_num_in_a_batch:
                    self.beam3_seq_start_positions.append(
                        [self.beam3_seq_start_positions[-1][-1]])
                    sub_seq_id = 0
                    seq_id += 1
                    sub_seq_count = len(self.beam2_seq_start_positions[
                        seq_id]) - 1
        assert (
            self.beam3_seq_start_positions[-1][-1] == self.end_scores.shape[0])

    def _build_seq_info_for_each_beam(self):
        self._build_beam1_seq_info()
        self._build_beam2_seq_info()
        self._build_beam3_seq_info()

    def _cal_ans_per_sample_in_a_batch(self):
        start_row = 0
        for seq in self.beam3_seq_start_positions:
            end_row = start_row + len(seq) - 1
            ans_count = np.sum(self.selected_ends[start_row:end_row, :] != -1.)

            self.ans_per_sample_in_a_batch.append(
                self.ans_per_sample_in_a_batch[-1] + ans_count)
            start_row = end_row

    def _get_valid_seleceted_ids(slef, mat):
        flattened = []
        height, width = mat.shape
        for i in range(height):
            for j in range(width):
                if mat[i][j] == -1.: break
                flattened.append([int(mat[i][j]), [i, j]])
        return flattened

    def decoding(self):
        self._build_seq_info_for_each_beam()
        self._cal_ans_per_sample_in_a_batch()

        seq_id = 0
        sub_seq_id = 0
        sub_seq_count = len(self.beam3_seq_start_positions[seq_id]) - 1

        sub_seq_num, beam_size = self.selected_ends.shape
        for i in xrange(sub_seq_num):
            seq_offset_in_batch = self.beam3_seq_start_positions[seq_id][
                sub_seq_id]
            for j in xrange(beam_size):
                end_pos = int(self.selected_ends[i][j])
                if end_pos == -1: break

                self.all_searched_ans.append({
                    "score": self.end_scores[seq_offset_in_batch + end_pos],
                    "sentence_pos": -1,
                    "start_span_pos": -1,
                    "end_span_pos": end_pos,
                    "parent_ids_in_prev_beam": i
                })

            sub_seq_id += 1
            if sub_seq_id == sub_seq_count:
                seq_id += 1
                if seq_id == len(self.beam3_seq_start_positions): break

                sub_seq_id = 0
                sub_seq_count = len(self.beam3_seq_start_positions[seq_id]) - 1

        assert len(self.all_searched_ans) == self.ans_per_sample_in_a_batch[-1]

        seq_id = 0
        sub_seq_id = 0
        sub_seq_count = len(self.beam2_seq_start_positions[seq_id]) - 1
        last_row_id = None

        starts = self._get_valid_seleceted_ids(self.selected_starts)
        for i, ans in enumerate(self.all_searched_ans):
            ans["start_span_pos"] = starts[ans["parent_ids_in_prev_beam"]][0]

            seq_offset_in_batch = (
                self.beam2_seq_start_positions[seq_id][sub_seq_id])
            ans["score"] += self.start_scores[(
                seq_offset_in_batch + ans["start_span_pos"])]
            ans["parent_ids_in_prev_beam"] = starts[ans[
                "parent_ids_in_prev_beam"]][1][0]

            if last_row_id and last_row_id != ans["parent_ids_in_prev_beam"]:
                sub_seq_id += 1

            if sub_seq_id == sub_seq_count:
                seq_id += 1
                if seq_id == len(self.beam2_seq_start_positions): break
                sub_seq_count = len(self.beam2_seq_start_positions[seq_id]) - 1
                sub_seq_id = 0
            last_row_id = ans["parent_ids_in_prev_beam"]

        offset_info = [0]
        for sen in self.beam1_seq_start_positions[:-1]:
            offset_info.append(offset_info[-1] + len(sen) - 1)
        sen_ids = self._get_valid_seleceted_ids(self.selected_sentences)
        for ans in self.all_searched_ans:
            ans["sentence_pos"] = sen_ids[ans["parent_ids_in_prev_beam"]][0]
            row_id = ans["parent_ids_in_prev_beam"] / beam_size
            offset = offset_info[row_id - 1] if row_id else 0
            ans["score"] += self.sentence_scores[offset + ans["sentence_pos"]]

        for i in range(len(self.ans_per_sample_in_a_batch) - 1):
            start_pos = self.ans_per_sample_in_a_batch[i]
            end_pos = self.ans_per_sample_in_a_batch[i + 1]

            for ans in sorted(
                    self.all_searched_ans[start_pos:end_pos],
                    key=lambda x: x["score"],
                    reverse=True):
                self.final_ans[i].append({
                    "score": ans["score"],
                    "label": [
                        ans["sentence_pos"], ans["start_span_pos"],
                        ans["end_span_pos"]
                    ]
                })

        return self.final_ans
