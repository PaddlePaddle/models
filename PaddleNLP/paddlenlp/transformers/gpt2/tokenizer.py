# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unicodedata
import json
import sentencepiece
import jieba

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize,\
    _is_whitespace, _is_control, _is_punctuation

__all__ = [
    'BasicTokenizer', 'GPT2Tokenizer', 'WordpieceTokenizer',
    'GPT2ChineseTokenizer'
]


class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.
    Args:
        vocab (Vocab|dict): Vocab of the word piece tokenizer.
        unk_token (str):  A specific token to replace all unkown tokens.
        max_input_chars_per_word (int):  If a word's length is more than
            max_input_chars_per_word, it will be dealt as unknown word.
            Default: 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.
        Returns:
            list (str): A list of wordpiece tokens.
        Example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class GPT2ChineseTokenizer(PretrainedTokenizer):
    """
    Constructs a GPT2 Chinese tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "model_file": "sentencepiece.model"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt2-base-cn": "http://10.255.129.12:8829/vocab.json",
        },
        "model_file": {
            "gpt2-base-cn": "http://10.255.129.12:8829/sentencepiece.model"
        }
    }
    pretrained_init_configuration = {"gpt2-base-cn": {"do_lower_case": True}, }

    def __init__(self,
                 vocab_file,
                 model_file,
                 do_lower_case=True,
                 max_len=512,
                 bod_id="<bod>",
                 eod_id="<eod>",
                 max_length=None):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [
            x.translate(self.translator) for x in jieba.cut(text, cut_all=False)
        ]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def convert_tokens_to_ids(self, text):
        res = self.tokenize(text)
        return res

    def convert_ids_to_tokens(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583',
                                                                    '\n')
        return text


class GPT2Tokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt2-medium-en": "http://10.255.129.12:8829/vocab.json",
        },
        "merges_file": {
            "gpt2-medium-en": "http://10.255.129.12:8829/merges.txt"
        }
    }
    pretrained_init_configuration = {
        "gpt2-medium-en": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 special_tokens=None,
                 max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {
            v: k
            for k, v in self.special_tokens.items()
        }
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i +
                                                                   1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and
                                       isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".
                format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text
