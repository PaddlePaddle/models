#coding=utf-8

import sys
import argparse
import distutils.util
import gzip

import paddle.v2 as paddle
from model import conv_seq2seq
from beamsearch import BeamSearch
import reader


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Convolutional Seq2Seq")
    parser.add_argument(
        '--infer_data_path',
        type=str,
        required=True,
        help="Path of the dataset for inference")
    parser.add_argument(
        '--src_dict_path',
        type=str,
        required=True,
        help='Path of the source dictionary')
    parser.add_argument(
        '--trg_dict_path',
        type=str,
        required=True,
        help='path of the target dictionary')
    parser.add_argument(
        '--enc_blocks', type=str, help='Convolution blocks of the encoder')
    parser.add_argument(
        '--dec_blocks', type=str, help='Convolution blocks of the decoder')
    parser.add_argument(
        '--emb_size',
        type=int,
        default=512,
        help='Dimension of word embedding. (default: %(default)s)')
    parser.add_argument(
        '--pos_size',
        type=int,
        default=200,
        help='Total number of the position indexes. (default: %(default)s)')
    parser.add_argument(
        '--drop_rate',
        type=float,
        default=0.,
        help='Dropout rate. (default: %(default)s)')
    parser.add_argument(
        "--use_gpu",
        default=False,
        type=distutils.util.strtobool,
        help="Use gpu or not. (default: %(default)s)")
    parser.add_argument(
        "--trainer_count",
        default=1,
        type=int,
        help="Trainer number. (default: %(default)s)")
    parser.add_argument(
        '--max_len',
        type=int,
        default=100,
        help="The maximum length of the sentence to be generated. (default: %(default)s)"
    )
    parser.add_argument(
        "--beam_size",
        default=1,
        type=int,
        help="Beam search width. (default: %(default)s)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path. (default: %(default)s)")
    return parser.parse_args()


def to_sentence(seq, dictionary):
    raw_sentence = [dictionary[id] for id in seq]
    sentence = " ".join(raw_sentence)
    return sentence


def infer(infer_data_path,
          src_dict_path,
          trg_dict_path,
          model_path,
          enc_conv_blocks,
          dec_conv_blocks,
          emb_dim=512,
          pos_size=200,
          drop_rate=0.,
          max_len=100,
          beam_size=1):
    """
    Inference.

    :param infer_data_path: The path of the data for inference.
    :type infer_data_path: str
    :param src_dict_path: The path of the source dictionary.
    :type src_dict_path: str
    :param trg_dict_path: The path of the target dictionary.
    :type trg_dict_path: str
    :param model_path: The path of a trained model.
    :type model_path: str
    :param enc_conv_blocks: The scale list of the encoder's convolution blocks. And each element of
                            the list contains output dimension and context length of the corresponding
                            convolution block.
    :type enc_conv_blocks: list of tuple
    :param dec_conv_blocks: The scale list of the decoder's convolution blocks. And each element of
                            the list contains output dimension and context length of the corresponding
                            convolution block.
    :type dec_conv_blocks: list of tuple
    :param emb_dim: The dimension of the embedding vector.
    :type emb_dim: int
    :param pos_size: The total number of the position indexes, which means
                     the maximum value of the index is pos_size - 1.
    :type pos_size: int
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param max_len: The maximum length of the sentence to be generated.
    :type max_len: int
    :param beam_size: The width of beam search.
    :type beam_size: int
    """
    # load dict
    src_dict = reader.load_dict(src_dict_path)
    trg_dict = reader.load_dict(trg_dict_path)
    src_dict_size = src_dict.__len__()
    trg_dict_size = trg_dict.__len__()

    prob = conv_seq2seq(
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        pos_size=pos_size,
        emb_dim=emb_dim,
        enc_conv_blocks=enc_conv_blocks,
        dec_conv_blocks=dec_conv_blocks,
        drop_rate=drop_rate,
        is_infer=True)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

    padding_list = [context_len - 1 for (size, context_len) in dec_conv_blocks]
    padding_num = reduce(lambda x, y: x + y, padding_list)
    infer_reader = reader.data_reader(
        data_file=infer_data_path,
        src_dict=src_dict,
        trg_dict=trg_dict,
        pos_size=pos_size,
        padding_num=padding_num)

    inferer = paddle.inference.Inference(
        output_layer=prob, parameters=parameters)

    searcher = BeamSearch(
        inferer=inferer,
        trg_dict=trg_dict,
        pos_size=pos_size,
        padding_num=padding_num,
        max_len=max_len,
        beam_size=beam_size)

    reverse_trg_dict = reader.get_reverse_dict(trg_dict)
    for i, raw_data in enumerate(infer_reader()):
        infer_data = [raw_data[0], raw_data[1]]
        result = searcher.search_one_sample(infer_data)
        sentence = to_sentence(result, reverse_trg_dict)
        print sentence
        sys.stdout.flush()
    return


def main():
    args = parse_args()
    enc_conv_blocks = eval(args.enc_blocks)
    dec_conv_blocks = eval(args.dec_blocks)

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    infer(
        infer_data_path=args.infer_data_path,
        src_dict_path=args.src_dict_path,
        trg_dict_path=args.trg_dict_path,
        model_path=args.model_path,
        enc_conv_blocks=enc_conv_blocks,
        dec_conv_blocks=dec_conv_blocks,
        emb_dim=args.emb_size,
        pos_size=args.pos_size,
        drop_rate=args.drop_rate,
        max_len=args.max_len,
        beam_size=args.beam_size)


if __name__ == '__main__':
    main()
