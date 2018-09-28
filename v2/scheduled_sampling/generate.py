import gzip
import argparse
import distutils.util
import paddle.v2 as paddle

from network_conf import seqToseq_net


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Scheduled Sampling")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="The path for trained model to load.")
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='The width of beam expansion. (default: %(default)s)')
    parser.add_argument(
        "--use_gpu",
        type=distutils.util.strtobool,
        default=False,
        help="Use gpu or not. (default: %(default)s)")
    parser.add_argument(
        "--trainer_count",
        type=int,
        default=1,
        help="Trainer number. (default: %(default)s)")

    return parser.parse_args()


def generate(gen_data, dict_size, model_path, beam_size):
    beam_gen = seqToseq_net(dict_size, dict_size, beam_size, is_generating=True)

    with gzip.open(model_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    # prob is the prediction probabilities, and id is the prediction word.
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=gen_data,
        field=['prob', 'id'])

    # get the dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

    # the delimited element of generated sequences is -1,
    # the first element of each generated sequence is the sequence length
    seq_list = []
    seq = []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    for i in xrange(gen_num):
        print "\n*******************************************************\n"
        print "src:", ' '.join([src_dict.get(w) for w in gen_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


if __name__ == '__main__':
    args = parse_args()

    dict_size = 30000

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    # use the first 3 samples for generation
    gen_creator = paddle.dataset.wmt14.gen(dict_size)
    gen_data = []
    gen_num = 3
    for item in gen_creator():
        gen_data.append((item[0], ))
        if len(gen_data) == gen_num:
            break

    generate(
        gen_data,
        dict_size=dict_size,
        model_path=args.model_path,
        beam_size=args.beam_size)
