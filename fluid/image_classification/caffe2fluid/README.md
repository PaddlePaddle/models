### Caffe2Fluid
This tool is used to convert a Caffe model to Fluid model

### Howto
1, Prepare caffepb.py in ./proto if your python has no 'pycaffe' module, two options provided here:
- generate it from caffe.proto using protoc
    bash ./proto/compile.sh

- download one from github directly
    cd proto/ && wget https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/caffe/caffepb.py

2, Convert the caffe model to fluid model
- generate fluid code and weight file
    python convert.py alexnet.prototxt \
        --caffemodel alexnet.caffemodel \
        --data-output-path alexnet.npy \
        --code-output-path alexnet.py

- save weights as fluid model file
    python alexnet.py alexnet.npy ./fluid_model

4, Use the converted model to predict
- see more detail info in 'examples/imagenet/run.sh'

5, compare the inference results with caffe
- see more detail info in 'examples/imagenet/diff.sh'

### Tested models
- Lenet

- ResNets:(ResNet-50, ResNet-101, ResNet-152)
[model addr](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

- GoogleNet:
[model addr](https://gist.github.com/jimmie33/7ea9f8ac0da259866b854460f4526034)

- VGG:
[model addr](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)

- AlexNet:
[model addr](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

### Notes
Some of this code come from here: https://github.com/ethereon/caffe-tensorflow
