import copy
import collections
import json
import os
import warnings

from paddle.io import Dataset


class RobustExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = False


class DuRobust(Dataset):
    def __init__(self, segment='train', root=None, **kwargs):

        if segment == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self.full_path = os.path.join(root, segment + '.json')
        self._read()

    def transform(self, fn):
        self.features = fn(self.examples)

    def _read(self):
        with open(self.full_path, "r", encoding="utf8") as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                    c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None

                    if self.is_training:
                        if len(qa["answers"]) != 1:
                            raise ValueError(
                                "For training, each question should have exactly 1 answer."
                            )

                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]

                    else:
                        orig_answer_text = []

                        answers = qa["answers"]
                        for answer in answers:
                            orig_answer_text.append(answer["text"])

                    example = RobustExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position)

                    examples.append(example)

        self.examples = examples

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        if self.is_training:
            return feature.input_ids, feature.segment_ids, feature.unique_id, feature.start_position, feature.end_position
        else:
            return feature.input_ids, feature.segment_ids, feature.unique_id
