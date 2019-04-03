### Caffe2Fluid
This tool is used to convert a Caffe model to a Fluid model

### Key Features
1. Convert caffe model to fluid model with codes of defining a network(useful for re-training)

2. Pycaffe is not necessary when just want convert model without do caffe-inference

3. Caffe's customized layers convertion also be supported by extending this tool

4. A bunch of tools in `examples/imagenet/tools` are provided to compare the difference

### HowTo
1. Prepare `caffepb.py` in `./proto` if your python has no `pycaffe` module, two options provided here:
    - Generate pycaffe from caffe.proto
        ```
        bash ./proto/compile.sh
        ```

    - Download one from github directly
        ```
        cd proto/ && wget https://raw.githubusercontent.com/ethereon/caffe-tensorflow/master/kaffe/caffe/caffepb.py
        ```

2. Convert the Caffe model to Fluid model
   - Generate fluid code and weight file
       ```
       python convert.py alexnet.prototxt \
               --caffemodel alexnet.caffemodel \
               --data-output-path alexnet.npy \
               --code-output-path alexnet.py
       ```

   - Save weights as fluid model file
       ```
       # only infer the last layer's result
       python alexnet.py alexnet.npy ./fluid
       # infer these 2 layer's result
       python alexnet.py alexnet.npy ./fluid fc8,prob
       ```

3. Use the converted model to infer
    - See more details in `examples/imagenet/tools/run.sh`

4. Compare the inference results with caffe
    - See more details in `examples/imagenet/tools/diff.sh`

### How to convert custom layer
1. Implement your custom layer in a file under `kaffe/custom_layers`, eg: mylayer.py
    - Implement ```shape_func(input_shape, [other_caffe_params])``` to calculate the output shape
    - Implement ```layer_func(inputs, name, [other_caffe_params])``` to construct a fluid layer
    - Register these two functions ```register(kind='MyType', shape=shape_func, layer=layer_func)```
    - Notes: more examples can be found in `kaffe/custom_layers`

2. Add ```import mylayer``` to  `kaffe/custom_layers/\_\_init__.py`

3. Prepare your pycaffe as your customized version(same as previous env prepare)
    - (option1) replace `proto/caffe.proto` with your own caffe.proto and compile it
    - (option2) change your `pycaffe` to the customized version

4. Convert the Caffe model to Fluid model

5. Set env $CAFFE2FLUID_CUSTOM_LAYERS to the parent directory of 'custom_layers'
   ```
   export CAFFE2FLUID_CUSTOM_LAYERS=/path/to/caffe2fluid/kaffe
   ```

6. Use the converted model when loading model in `xxxnet.py` and `xxxnet.npy`(no need if model is already in `fluid/model` and `fluid/params`)

### Tested models
- Lenet:
[model addr](https://github.com/ethereon/caffe-tensorflow/blob/master/examples/mnist)

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
