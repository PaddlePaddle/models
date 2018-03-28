import click
import gzip

import paddle.v2 as paddle
from network_conf import Model
from reader import DataGenerator
from decoder import ctc_greedy_decoder
from utils import get_file_list, load_dict, load_reverse_dict


def infer_batch(inferer, test_batch, labels, reversed_char_dict):
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(0, len(test_batch))
    ]
    results = []
    # Best path decode.
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=reversed_char_dict)
        results.append(output_transcription)

    for result, label in zip(results, labels):
        print("\nOutput Transcription: %s\nTarget Transcription: %s" %
              (result, label))


@click.command('infer')
@click.option(
    "--model_path", type=str, required=True, help=("The path of saved model."))
@click.option(
    "--image_shape",
    type=str,
    required=True,
    help=("The fixed size for image dataset (format is like: '173,46')."))
@click.option(
    "--batch_size",
    type=int,
    default=10,
    help=("The number of examples in one batch (default: 10)."))
@click.option(
    "--label_dict_path",
    type=str,
    required=True,
    help=("The path of label dictionary. "))
@click.option(
    "--infer_file_list_path",
    type=str,
    required=True,
    help=("The path of the file which contains "
          "path list of image files for inference."))
def infer(model_path, image_shape, batch_size, label_dict_path,
          infer_file_list_path):

    image_shape = tuple(map(int, image_shape.split(',')))
    infer_file_list = get_file_list(infer_file_list_path)

    char_dict = load_dict(label_dict_path)
    reversed_char_dict = load_reverse_dict(label_dict_path)
    dict_size = len(char_dict)
    data_generator = DataGenerator(char_dict=char_dict, image_shape=image_shape)

    paddle.init(use_gpu=True, trainer_count=1)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    model = Model(dict_size, image_shape, is_infer=True)
    inferer = paddle.inference.Inference(
        output_layer=model.log_probs, parameters=parameters)

    test_batch = []
    labels = []
    for i, (image, label
            ) in enumerate(data_generator.infer_reader(infer_file_list)()):
        test_batch.append([image])
        labels.append(label)
        if len(test_batch) == batch_size:
            infer_batch(inferer, test_batch, labels, reversed_char_dict)
            test_batch = []
            labels = []
        if test_batch:
            infer_batch(inferer, test_batch, labels, reversed_char_dict)


if __name__ == "__main__":
    infer()
