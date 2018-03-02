### Caffe2Fluid
This tool is used to convert a Caffe model to Fluid model

### Howto
1, Prepare caffepb.py in ./proto, two options provided
    1) generate it from caffe.proto using protoc
        bash ./proto/compile.sh

    2) download one from github directly
        cd proto/ && wget https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/caffe/caffepb.py

2, Convert the caffe model using convert.py which will generate a python code and weight(in .npy)

3, Use the converted model to predict
    see more detail info in 'tests/lenet/README.md'


### Supported models
- Lenet on mnist dataset

- ResNets:(ResNet-50, ResNet-101, ResNet-152)
    model addrs:(https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

### Notes
Some of this code come from here: https://github.com/ethereon/caffe-tensorflow
