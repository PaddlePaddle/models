import os
import math


class KenLMModel(object):
    """
    Wrapper for KenLM language model.
    You should install KenLM python interface first.

    .. code-block:: python
    
        pip install https://github.com/kpu/kenlm/archive/master.zip

    Please refer to **Scalable Modified Kneser-Ney Language Model Estimation** for
    more details about KenLM
    """

    def __init__(self,
                 model_path,
                 unk_id,
                 bos_id,
                 eos_id,
                 id_str_dict,
                 verbose=False):
        """
        Initialize variables and load model.

        :param model_path: Path of language model
        :type model_path: str
        :param unk_id: Identifier for OOV
        :type unk_id: int
        :param bos_id: Identifier for start token
        :type bos_id: int
        :param eos_id: Identifier for end token
        :type eos_id: int
        :param id_str_dict: Dictionary mapping id to word
        :type id_str_dict: dict
        :param verbose: Whether print debug information
        :type verbose: bool
        """
        assert unk_id in id_str_dict, 'unk_id must be in id_str_dict'
        assert bos_id in id_str_dict, 'bos_id must be in id_str_dict'
        assert eos_id in id_str_dict, 'eos_id must be in id_str_dict'
        self._model_path = model_path
        self._unk_id = unk_id
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._id_str_dict = id_str_dict
        self._verbose = verbose

        self._load_model()
        if self._verbose:
            print("Load model done.")

    def _load_model(self):
        import kenlm
        self._model = kenlm.LanguageModel(self._model_path)

    def score_sentence_ids(self, id_list):
        """
        Get quality score for input sentence which represented by id list.
        In function **score** of KenLM, input sentence is treated as completed
        sentence which includes start token and end token. We will set bos flag
        to true and input id list should never include start token otherwise the
        start token will be dropped. We will set eos to false, so you should append
        end token explicitly if the input sentence has been completed. 

        :param id_list: Id list of word.
        :param id_str_dict: list
        """
        assert len(id_list) > 0, 'invalid id list'
        eos = False  # Always false, user should pad eos id explicitly
        bos = True
        if id_list[0] == self._bos_id:
            id_list = id_list[1:]  # Never include start token

        char_list = []
        for str_id in id_list:
            assert str_id in self._id_str_dict, '%d not in dictionary' % str_id
            if str_id == self._eos_id:
                char_list.append('</s>')
            elif str_id == self._unk_id:
                char_list.append('<unk>')
            else:
                char_list.append(self._id_str_dict[str_id])

        sentence = ' '.join(char_list)
        score = self._model.score(sentence, bos=bos, eos=eos)
        return math.pow(10, score)


if __name__ == '__main__':
    id_str_dict = {}
    id_str_dict[0] = '<unk>'
    id_str_dict[1] = '<s>'
    id_str_dict[2] = '</s>'
    id_str_dict[3] = 'Hello'
    id_str_dict[4] = 'world'

    kenLM = KenLMModel(
        model_path='1Billion.klm',
        unk_id=0,
        bos_id=1,
        eos_id=2,
        id_str_dict=id_str_dict,
        verbose=False)

    print kenLM.score_sentence_ids([3, 4])  # Hello world: 2.51940257717e-08
    print kenLM.score_sentence_ids(
        [3, 3, 4])  # Hello Hello world: 5.28953240539e-14
    print kenLM.score_sentence_ids(
        [3, 0, 4])  # Hello <unk> world: 9.03021339418e-18
