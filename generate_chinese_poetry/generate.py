import os
import sys
import gzip
import logging
import numpy as np
import click

import reader
import paddle.v2 as paddle
from paddle.v2.layer import parse_network
from network_conf import encoder_decoder_network

logger = logging.getLogger("paddle")
logger.setLevel(logging.WARNING)


def infer_a_batch(inferer, test_batch, beam_size, id_to_text, fout):
    beam_result = inferer.infer(input=test_batch, field=["prob", "id"])
    gen_sen_idx = np.where(beam_result[1] == -1)[0]
    assert len(gen_sen_idx) == len(test_batch) * beam_size, ("%d vs. %d" % (
        len(gen_sen_idx), len(test_batch) * beam_size))

    start_pos, end_pos = 1, 0
    for i, sample in enumerate(test_batch):
        fout.write("%s\n" % (
            " ".join([id_to_text[w] for w in sample[0][1:-1]])
        ))  # skip the start and ending mark when print the source sentence
        for j in xrange(beam_size):
            end_pos = gen_sen_idx[i * beam_size + j]
            fout.write("%s\n" % ("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                id_to_text[w] for w in beam_result[1][start_pos:end_pos - 1]))))
            start_pos = end_pos + 2
        fout.write("\n")
        fout.flush


@click.command("generate")
@click.option(
    "--model_path",
    default="",
    help="The path of the trained model for generation.")
@click.option(
    "--word_dict_path", required=True, help="The path of word dictionary.")
@click.option(
    "--test_data_path",
    required=True,
    help="The path of input data for generation.")
@click.option(
    "--batch_size",
    default=1,
    help="The number of testing examples in one forward pass in generation.")
@click.option(
    "--beam_size", default=5, help="The beam expansion in beam search.")
@click.option(
    "--save_file",
    required=True,
    help="The file path to save the generated results.")
@click.option(
    "--use_gpu", default=False, help="Whether to use GPU in generation.")
def generate(model_path, word_dict_path, test_data_path, batch_size, beam_size,
             save_file, use_gpu):
    assert os.path.exists(model_path), "The given model does not exist."
    assert os.path.exists(test_data_path), "The given test data does not exist."

    with gzip.open(model_path, "r") as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    id_to_text = {}
    assert os.path.exists(
        word_dict_path), "The given word dictionary path does not exist."
    with open(word_dict_path, "r") as f:
        for i, line in enumerate(f):
            id_to_text[i] = line.strip().split("\t")[0]

    paddle.init(use_gpu=use_gpu, trainer_count=1)
    beam_gen = encoder_decoder_network(
        word_count=len(id_to_text),
        emb_dim=512,
        encoder_depth=3,
        encoder_hidden_dim=512,
        decoder_depth=3,
        decoder_hidden_dim=512,
        bos_id=0,
        eos_id=1,
        max_length=9,
        beam_size=beam_size,
        is_generating=True)

    inferer = paddle.inference.Inference(
        output_layer=beam_gen, parameters=parameters)

    test_batch = []
    with open(save_file, "w") as fout:
        for idx, item in enumerate(
                reader.gen_reader(test_data_path, word_dict_path)()):
            test_batch.append([item])
            if len(test_batch) == batch_size:
                infer_a_batch(inferer, test_batch, beam_size, id_to_text, fout)
                test_batch = []

        if len(test_batch):
            infer_a_batch(inferer, test_batch, beam_size, id_to_text, fout)
            test_batch = []


if __name__ == "__main__":
    generate()
