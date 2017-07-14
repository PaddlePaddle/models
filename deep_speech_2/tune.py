"""Parameters tuning for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import argparse
import gzip
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import deep_speech2
from decoder import *
from lm.lm_scorer import LmScorer
from error_rate import wer
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--batch_size",
    default=100,
    type=int,
    help="Minibatch size for evaluation. (default: %(default)s)")
parser.add_argument(
    "--num_conv_layers",
    default=2,
    type=int,
    help="Convolution layer number. (default: %(default)s)")
parser.add_argument(
    "--num_rnn_layers",
    default=3,
    type=int,
    help="RNN layer number. (default: %(default)s)")
parser.add_argument(
    "--rnn_layer_size",
    default=512,
    type=int,
    help="RNN layer cell number. (default: %(default)s)")
parser.add_argument(
    "--use_gpu",
    default=True,
    type=distutils.util.strtobool,
    help="Use gpu or not. (default: %(default)s)")
parser.add_argument(
    "--num_threads_data",
    default=multiprocessing.cpu_count(),
    type=int,
    help="Number of cpu threads for preprocessing data. (default: %(default)s)")
parser.add_argument(
    "--num_processes_beam_search",
    default=multiprocessing.cpu_count(),
    type=int,
    help="Number of cpu processes for beam search. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='datasets/manifest.dev',
    type=str,
    help="Manifest path for decoding. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='checkpoints/params.latest.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--language_model_path",
    default="lm/data/common_crawl_00.prune01111.trie.klm",
    type=str,
    help="Path for language model. (default: %(default)s)")
parser.add_argument(
    "--alpha_from",
    default=0.22,
    type=float,
    help="Where alpha starts from. (default: %(default)f)")
parser.add_argument(
    "--num_alphas",
    default=10,
    type=int,
    help="Number of candidate alphas. (default: %(default)d)")
parser.add_argument(
    "--alpha_to",
    default=0.40,
    type=float,
    help="Where alpha ends with. (default: %(default)f)")
parser.add_argument(
    "--beta_from",
    default=0.05,
    type=float,
    help="Where beta starts from. (default: %(default)f)")
parser.add_argument(
    "--num_betas",
    default=7,
    type=float,
    help="Number of candidate betas. (default: %(default)d)")
parser.add_argument(
    "--beta_to",
    default=0.35,
    type=float,
    help="Where beta ends with. (default: %(default)f)")
parser.add_argument(
    "--cutoff_prob",
    default=0.99,
    type=float,
    help="The cutoff probability of pruning"
    "in beam search. (default: %(default)f)")
args = parser.parse_args()


def tune():
    """Tune parameters alpha and beta on one minibatch."""

    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")

    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}',
        num_threads=args.num_threads_data)

    # create network config
    # paddle.data_type.dense_array is used for variable batch input.
    # The size 161 * 161 is only an placeholder value and the real shape
    # of input batch data will be induced during training.
    audio_data = paddle.layer.data(
        name="audio_spectrogram", type=paddle.data_type.dense_array(161 * 161))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(data_generator.vocab_size))
    output_probs = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=True)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_filepath))

    # prepare infer data
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.decode_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=1,
        sortagrad=False,
        shuffle_method=None)

    # define inferer
    inferer = paddle.inference.Inference(
        output_layer=output_probs, parameters=parameters)

    # create grid for search
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas
                   for beta in cand_betas]

    # external scorer
    ext_scorer = LmScorer(args.alpha_from, args.beta_from,
                          args.language_model_path)

    wer_sum = [0.0 for i in xrange(len(params_grid))]
    wer_counter = [0 for i in xrange(len(params_grid))]
    ave_wer = [0.0 for i in xrange(len(params_grid))]
    num_batches = 0

    ## incremental tuning batch by batch
    for infer_data in batch_reader():
        # run inference
        infer_results = inferer.infer(input=infer_data)
        num_steps = len(infer_results) // len(infer_data)
        probs_split = [
            infer_results[i * num_steps:(i + 1) * num_steps]
            for i in xrange(0, len(infer_data))
        ]
        # target transcription
        target_transcription = [
            ''.join([
                data_generator.vocab_list[index] for index in infer_data[i][1]
            ]) for i, probs in enumerate(probs_split)
        ]

        # grid search on current batch
        for index, (alpha, beta) in enumerate(params_grid):
            # reset scorer
            ext_scorer.reset_params(alpha, beta)
            beam_search_results = ctc_beam_search_decoder_batch(
                probs_split=probs_split,
                vocabulary=data_generator.vocab_list,
                beam_size=args.beam_size,
                blank_id=len(data_generator.vocab_list),
                num_processes=args.num_processes_beam_search,
                ext_scoring_func=ext_scorer,
                cutoff_prob=args.cutoff_prob, )
            for i, beam_search_result in enumerate(beam_search_results):
                wer_sum[index] += wer(target_transcription[i],
                                      beam_search_result[0][1])
                wer_counter[index] += 1
            ave_wer[index] = wer_sum[index] / wer_counter[index]
            print("alpha = %f, beta = %f,  WER = %f" %
                  (alpha, beta, ave_wer[index]))

        # output tuning result til current batch
        ave_wer_min = min(ave_wer)
        min_index = ave_wer.index(ave_wer_min)
        print("Finish batch %d, alpha_opt = %f, beta_opt = %f, WER_opt = %f\n" %
              (num_batches, params_grid[min_index][0],
               params_grid[min_index][1], ave_wer_min))
        num_batches += 1


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    tune()


if __name__ == '__main__':
    main()
