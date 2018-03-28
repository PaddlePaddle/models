import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from load_model import load_param
from utility import get_feeder_data
from crnn_ctc_model import ctc_infer
import ctc_reader
import dummy_reader


def load_parameter(place):
    params = load_param('./name.map', './data/model/results/pass-00062/')
    for name in params:
        print "param: %s" % name
        t = fluid.global_scope().find_var(name).get_tensor()
        t.set(params[name], place)


def inference(infer=ctc_infer, data_reader=dummy_reader):
    """OCR inference"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    sequence, tmp = infer(images, num_classes)
    fluid.layers.Print(tmp)
    # data reader
    test_reader = data_reader.test()
    # prepare environment
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    load_parameter(place)

    for data in test_reader():
        result = exe.run(fluid.default_main_program(),
                         feed=get_feeder_data(
                             data, place, need_label=False),
                         fetch_list=[tmp])
        print "result: %s" % (list(result[0].flatten()), )


def main():
    inference(data_reader=ctc_reader)


if __name__ == "__main__":
    main()
