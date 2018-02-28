### convert lenet model from caffe format into paddle format(fluid api)

### howto
    0, prepare your caffepb.py

    1, download a lenet caffe-model
        lenet_iter_10000.caffemodel
            download address: https://github.com/ethereon/caffe-tensorflow/raw/master/examples/mnist/lenet_iter_10000.caffemodel
            md5: cbec75c1c374b6c1981c4a1eb024ae01  

        lenet.prototxt
            download address: https://raw.githubusercontent.com/BVLC/caffe/master/examples/mnist/lenet.prototxt
            md5: 27384af843338ab90b00c8d1c81de7d5


    2, convert this model(make sure caffepb.py is ready in ../../proto)
        * convert to npy format
            bash ./convert.sh lenet.prototxt lenet.caffemodel lenet.py lenet.npy

        * save to fluid format(optional)
            bash ./convert.sh lenet.prototxt lenet.caffemodel lenet.py lenet.npy && python ./lenet.py ./lenet.npy ./fluid.model

    4, use this new model(paddle installed in this python)
        * use fluid format
            python ./predict.py ./fluid.model

        * use npy format
            python ./predict.py ./lenet.npy
