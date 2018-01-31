### convert lenet model from caffe format into paddle format(fluid api)

### howto
    1, download a lenet model here: https://github.com/ethereon/caffe-tensorflow/tree/master/examples/mnist

    2, convert this model
        bash ./convert.sh lenet.prototxt lenet.caffemodel lenet.py lenet.npy

    3, use this new model
        python ./predict.py
