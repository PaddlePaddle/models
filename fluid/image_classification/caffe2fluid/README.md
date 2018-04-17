### Caffe2Fluid
This tool is used to convert a Caffe model to a Fluid model

### HowTo
1. Prepare caffepb.py in ./proto if your python has no 'pycaffe' module, two options provided here:
- Generate pycaffe from caffe.proto
  <pre><code>bash ./proto/compile.sh</code></pre>

- download one from github directly
  <pre><code>cd proto/ && wget https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/caffe/caffepb.py</code></pre>

2. Convert the Caffe model to Fluid model
- generate fluid code and weight file
  <pre><code>python convert.py alexnet.prototxt \
        --caffemodel alexnet.caffemodel \
        --data-output-path alexnet.npy \
        --code-output-path alexnet.py</code></pre>

- save weights as fluid model file
  <pre><code>python alexnet.py alexnet.npy ./fluid_model</code></pre>

3. Use the converted model to infer
- see more details in '*examples/imagenet/run.sh*'

4. compare the inference results with caffe
- see more details in '*examples/imagenet/diff.sh*'

### How to convert custom layer
1. implement your custom layer in a file under '*kaffe/custom_layers*', eg: mylayer.py
- implement a 'shape_func(input_shape, [other_caffe_params])' to calculate the output shape
- implement a 'layer_func(inputs, name, [other_caffe_params])' to construct a fluid layer
- register these two functions using '*register*'
- notes: more examples can be found in '*kaffe/custom_layers*'

2. edit '*kaffe/custom_layers/\_\_init__.py*' to add your custom layer after line 5
   <pre><code>import mylayer</code></pre>

3. prepare your pycaffe as your customized version(same as previous env prepare)
- (option1) replace 'proto/caffe.proto' with your own caffe.proto and compile it
- (option2) change your pycaffe to the customized version

4. convert the model
5. set env $CAFFE2FLUID_CUSTOM_LAYERS to the directory of 'custom_layers'
   <pre><code>export CAFFE2FLUID_CUSTOM_LAYERS=/path/to/caffe2fluid/kaffe</code></pre>
6. use the converted model when loading model in 'xxx.py' and 'xxx.npy'(no need if model is already in 'fluid/model' and 'fluid/params')

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
Some of this code come from here: [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
