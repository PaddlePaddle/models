### caffe2fluid
    this tool is used to convert a caffe-model to paddle-model(fluid api)

### howto
    1, prepare caffepb.py in ./proto, two options provided
        option 1: generate it from caffe.proto using protoc
            bash ./proto/compile.sh

        option2: download one from github directly
            cd proto/ && wget https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/caffe/caffepb.py

    2, convert you caffe model using convert.py which will generate a python code and weight(in .npy)

    3, use the converted model to predict

    (see more detail info in 'tests/lenet/README.md')


### supported models
    lenet on mnist dataset

    resnets:(resnet50, resnet101, resnet152)
        model addrs:(https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

### notes
    some of this code come from here: https://github.com/ethereon/caffe-tensorflow
