import paddle.v2 as paddle
import paddle.fluid as fluid
from load_model import load_param
from utility import get_feeder_data
from crnn_ctc_model import ctc_eval
import ctc_reader
import dummy_reader


def load_parameter(place):
    params = load_param('./name.map', './data/model/results/pass-00062/')
    for name in params:
        print "param: %s" % name
        t = fluid.global_scope().find_var(name).get_tensor()
        t.set(params[name], place)


def evaluate(eval=ctc_eval, data_reader=dummy_reader):
    """OCR inference"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)
    evaluator, cost = eval(images, label, num_classes)

    # data reader
    test_reader = data_reader.test()
    # prepare environment
    place = fluid.CUDAPlace(0)
    #place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    print fluid.default_main_program()
    load_parameter(place)
    evaluator.reset(exe)
    count = 0
    for data in test_reader():
        count += 1
        print 'Process samples: %d\r' % (count, ),
        result, avg_distance, avg_seq_error = exe.run(
            fluid.default_main_program(),
            feed=get_feeder_data(data, place),
            fetch_list=[cost] + evaluator.metrics)
    avg_distance, avg_seq_error = evaluator.eval(exe)
    print "avg_distance: %s; avg_seq_error: %s" % (avg_distance, avg_seq_error)


def main():
    evaluate(data_reader=ctc_reader)


if __name__ == "__main__":
    main()
