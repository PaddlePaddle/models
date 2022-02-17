# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""A tokenizer for decoding wav2vec 2.0 preditions"""

import json
from itertools import groupby
from typing import Dict, List, Optional, Union

from paddle.utils import download
from paddleaudio.utils import get_logger

URL_BASE = 'https://bj.bcebos.com/paddleaudio/models/wav2vec2/'
logger = get_logger(__file__)


class Wav2Vec2Tokenizer():
    """
    Constructs a Wav2Vec2 tokenizer.

    This tokenizer used for decoding predictions into letters.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        The other arguments should be left unchanged for this version of wav2vec models.

    """
    model_input_names = ["input_values", "attention_mask"]

    def __init__(self,
                 vocab_file=None,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 word_delimiter_token="|",
                 do_lower_case=True,
                 verbose=False):
        if vocab_file is None:
            vocab_url = URL_BASE + 'config/vocab.json'
            vocab_file = download.get_weights_path_from_url(vocab_url)

        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        self._sep_token = None
        self._pad_token = pad_token
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._word_delimiter_token = word_delimiter_token

        self.verbose = verbose

        self._additional_special_tokens = []

        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}

        self.do_lower_case = do_lower_case

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def bos_token(self) -> str:
        """
        :obj:`str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None and self.verbose:
            logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        :obj:`str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None and self.verbose:
            logger.error("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        :obj:`str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        :obj:`str`: Separation token, to separate context and query in an input sequence. Log an error if used while
        not having been set.
        """
        if self._sep_token is None and self.verbose:
            logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None and self.verbose:
            logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        :obj:`str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the
        full depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None and self.verbose:
            logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        :obj:`str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while
        not having been set.
        """
        if self._mask_token is None and self.verbose:
            logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        :obj:`List[str]`: All the additional special tokens you may want to use. Log an error if used while not having
        been set.
        """
        if self._additional_special_tokens is None and self.verbose:
            logger.error(
                "Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def unk_token(self) -> str:
        """
        :obj:`str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def word_delimiter_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (:obj:`str`): The text to clean up.

        Returns:
            :obj:`str`: The cleaned-up string.
        """
        out_string = (out_string.replace(" .", ".").replace(" ?", "?").replace(
            " !", "!").replace(" ,", ",").replace(" ' ", "'").replace(
                " n't",
                "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(
                    " 've", "'ve").replace(" 're", "'re"))
        return out_string

    def convert_ids_to_tokens(
            self,
            ids: Union[int, List[int]],
            skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(
            filter(lambda token: token != self.pad_token, grouped_tokens))

        # replace delimiter token
        string = "".join([
            " " if token == self.word_delimiter_token else token
            for token in filtered_tokens
        ]).strip()

        if self.do_lower_case:
            string = string.lower()
        return string

    def decode(self,
               token_ids: List[int],
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True,
               **kwargs) -> str:
        """
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(result)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def convert_tokens_to_ids(
            self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    @property
    def all_special_ids(self) -> List[int]:
        """
        :obj:`List[int]`: List the ids of the special tokens(:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class
        attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
