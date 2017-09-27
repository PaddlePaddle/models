import logging
import argparse
import paddle.v2 as paddle
import gzip
from model import Model
from data_provider import get_file_list, AsciiDic, ImageDataset
from decoder import ctc_greedy_decoder


def infer(inferer, test_batch, labels):
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(0, len(test_batch))
    ]

    results = []
    # best path decode
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=AsciiDic().id2word())
        results.append(output_transcription)

    for result, label in zip(results, labels):
        print("\nOutput Transcription: %s\nTarget Transcription: %s" % (result,
                                                                        label))


if __name__ == "__main__":
    model_path = "model.ctc-pass-1-batch-150-test-10.2607016472.tar.gz"
    image_shape = "173,46"
    batch_size = 50
    infer_file_list = 'data/test_data/Challenge2_Test_Task3_GT.txt'
    image_shape = tuple(map(int, image_shape.split(',')))
    infer_generator = get_file_list(infer_file_list)

    dataset = ImageDataset(None, None, infer_generator, image_shape, True)

    paddle.init(use_gpu=True, trainer_count=4)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    model = Model(AsciiDic().size(), image_shape, is_infer=True)
    inferer = paddle.inference.Inference(
        output_layer=model.log_probs, parameters=parameters)

    test_batch = []
    labels = []
    for i, (image, label) in enumerate(dataset.infer()):
        test_batch.append([image])
        labels.append(label)
        if len(test_batch) == batch_size:
            infer(inferer, test_batch, labels)
            test_batch = []
            labels = []
        if test_batch:
            infer(inferer, test_batch, labels)
