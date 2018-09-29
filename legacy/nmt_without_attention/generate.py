import os
import logging
import numpy as np

from network_conf import seq2seq_net

logger = logging.getLogger("paddle")
logger.setLevel(logging.WARNING)


def infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict):
    beam_result = inferer.infer(input=test_batch, field=["prob", "id"])

    gen_sen_idx = np.where(beam_result[1] == -1)[0]
    assert len(gen_sen_idx) == len(test_batch) * beam_size

    start_pos, end_pos = 1, 0
    for i, sample in enumerate(test_batch):
        print(" ".join([
            src_dict[w] for w in sample[0][1:-1]
        ]))  # skip the start and ending mark when print the source sentence
        for j in xrange(beam_size):
            end_pos = gen_sen_idx[i * beam_size + j]
            print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                trg_dict[w] for w in beam_result[1][start_pos:end_pos])))
            start_pos = end_pos + 2
        print("\n")


def generate(source_dict_dim, target_dict_dim, model_path, beam_size,
             batch_size):
    """
    Sequence generation for NMT.

    :param source_dict_dim: size of source dictionary
    :type source_dict_dim: int
    :param target_dict_dim: size of target dictionary
    :type target_dict_dim: int
    :param model_path: path for inital model
    :type model_path: string
    :param beam_size: the expanson width in each generation setp
    :param beam_size: int
    :param batch_size: the number of training examples in one forward pass
    :param batch_size: int
    """

    assert os.path.exists(model_path), "trained model does not exist."

    # step 1: prepare dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(source_dict_dim)

    # step 2: load the trained model
    paddle.init(use_gpu=False, trainer_count=1)
    with gzip.open(model_path) as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    beam_gen = seq2seq_net(
        source_dict_dim,
        target_dict_dim,
        beam_size=beam_size,
        max_length=100,
        is_generating=True)
    inferer = paddle.inference.Inference(
        output_layer=beam_gen, parameters=parameters)

    # step 3: iterating over the testing dataset
    test_batch = []
    for idx, item in enumerate(paddle.dataset.wmt14.gen(source_dict_dim)()):
        test_batch.append([item[0]])
        if len(test_batch) == batch_size:
            infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict)
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, beam_size, src_dict, trg_dict)
        test_batch = []


if __name__ == "__main__":
    generate(
        source_dict_dim=30000,
        target_dict_dim=30000,
        batch_size=20,
        beam_size=3,
        model_path="models/nmt_without_att_params_batch_00100.tar.gz")
