import os
import gzip
import numpy as np

import paddle.v2 as paddle

from utils import logger, load_dict
from beam_search import BeamSearch
import config as conf
from network_conf import rnn_lm


def rnn_generate(gen_input_file, model_path, max_gen_len, beam_size,
                 word_dict_file):
    """
    use RNN model to generate sequences.

    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of "{word, id}",
                        "word" is string type , "id" is int type.
    :param num_words: the number of the words to generate.
    :type num_words: int
    :param beam_size: beam width.
    :type beam_size: int
    :return: save prediction results to output_file
    """

    assert os.path.exists(gen_input_file), "test file does not exist!"
    assert os.path.exists(model_path), "trained model does not exist!"
    assert os.path.exists(
        word_dict_file), "word dictionary file does not exist!"

    # load word dictionary
    word_2_ids = load_dict(word_dict_file)
    try:
        UNK_ID = word_2_ids["<unk>"]
    except KeyError:
        logger.fatal("the word dictionary must contain a <unk> token!")
        sys.exit(-1)

    # initialize paddle
    paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

    # load the trained model
    pred_words = rnn_lm(
        len(word_2_ids),
        conf.emb_dim,
        conf.hidden_size,
        conf.stacked_rnn_num,
        conf.rnn_type,
        is_infer=True)

    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))

    inferer = paddle.inference.Inference(
        output_layer=pred_words, parameters=parameters)

    generator = BeamSearch(inferer, word_dict_file, beam_size, max_gen_len)
    # generate text
    with open(conf.gen_file, "r") as fin, open(conf.gen_result, "w") as fout:
        for idx, line in enumerate(fin):
            fout.write("%d\t%s" % (idx, line))
            for gen_res in generator.gen_a_sentence([
                    word_2_ids.get(w, UNK_ID)
                    for w in line.lower().strip().split()
            ]):
                fout.write("%s\n" % gen_res)
            fout.write("\n")


if __name__ == "__main__":
    rnn_generate(conf.gen_file, conf.model_path, conf.max_gen_len,
                 conf.beam_size, conf.vocab_file)
