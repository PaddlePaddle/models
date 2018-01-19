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
        default=256,
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
        "--use_bn",
        default=False,
        type=distutils.util.strtobool,
        help="Use batch normalization or not. (default: %(default)s)")
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
        "--batch_size",
        default=1,
        type=int,
        help="Size of a mini-batch. (default: %(default)s)")
    parser.add_argument(
        "--beam_size",
        default=1,
        type=int,
        help="The width of beam expansion. (default: %(default)s)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path of trained model. (default: %(default)s)")
    parser.add_argument(
        "--is_show_attention",
        default=False,
        type=distutils.util.strtobool,
        help="Whether to show attention weight or not. (default: %(default)s)")
    return parser.parse_args()


def infer(infer_data_path,
          src_dict_path,
          trg_dict_path,
          model_path,
          enc_conv_blocks,
          dec_conv_blocks,
          emb_dim=256,
          pos_size=200,
          drop_rate=0.,
          use_bn=False,
          max_len=100,
          batch_size=1,
          beam_size=1,
          is_show_attention=False):
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
    :param use_bn: Whether to use batch normalization or not. False is the default value.
    :type use_bn: bool
    :param max_len: The maximum length of the sentence to be generated.
    :type max_len: int
    :param beam_size: The width of beam expansion.
    :type beam_size: int
    :param is_show_attention: Whether to show attention weight or not. False is the default value.
    :type is_show_attention: bool
    """
    # load dict
    src_dict = reader.load_dict(src_dict_path)
    trg_dict = reader.load_dict(trg_dict_path)
    src_dict_size = src_dict.__len__()
    trg_dict_size = trg_dict.__len__()

    prob, weight = conv_seq2seq(
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        pos_size=pos_size,
        emb_dim=emb_dim,
        enc_conv_blocks=enc_conv_blocks,
        dec_conv_blocks=dec_conv_blocks,
        drop_rate=drop_rate,
        with_bn=use_bn,
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

    if is_show_attention:
        attention_inferer = paddle.inference.Inference(
            output_layer=weight, parameters=parameters)
        for i, data in enumerate(infer_reader()):
            src_len = len(data[0])
            trg_len = len(data[2])
            attention_weight = attention_inferer.infer(
                [data], field='value', flatten_result=False)
            attention_weight = [
                weight.reshape((trg_len, src_len))
                for weight in attention_weight
            ]
            print attention_weight
            break
        return

    infer_data = []
    for i, raw_data in enumerate(infer_reader()):
        infer_data.append([raw_data[0], raw_data[1]])

    inferer = paddle.inference.Inference(
        output_layer=prob, parameters=parameters)

    searcher = BeamSearch(
        inferer=inferer,
        trg_dict=trg_dict,
        pos_size=pos_size,
        padding_num=padding_num,
        max_len=max_len,
        batch_size=batch_size,
        beam_size=beam_size)

    searcher.search(infer_data)
    return


def main():
    args = parse_args()
    enc_conv_blocks = eval(args.enc_blocks)
    dec_conv_blocks = eval(args.dec_blocks)

    sys.setrecursionlimit(10000)

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
        use_bn=args.use_bn,
        max_len=args.max_len,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        is_show_attention=args.is_show_attention)


if __name__ == '__main__':
    main()
