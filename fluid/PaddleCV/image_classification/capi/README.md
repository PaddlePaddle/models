# C-API inference application
The `infer_image_classification` application serves for running C-API based
inference benchmark for image classification models: ResNet50, SE-ResNeXt50
and MobileNet-v1.

# How to build C-API application
In order to build C-API inference application follow these three steps:
1. build paddle.
2. build paddle's target `fluid_lib_dist`.
3. build capi inference application.

Each one will be shortly described below.
## 1. Build paddle
Do it as you usually do it. In case you never did it, here are example instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DWITH_TIMER=OFF -DWITH_PROFILER=OFF -DWITH_FLUID_ONLY=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j <num_cpu_cores>
```
## 2. Build paddle's target `fluid_lib_dist`
While still staying in `/path/to/Paddle/build`, build the target `fluid_lib_dist`:
```
make -j <num_cpu_cores> fluid_lib_dist
```
Now a directory named `fluid_install_dir` should exist in the build directory.
Remember that path.

## 3. Build C-API inference application
Go to the location of this README.md (... models/fluid/PaddleCV/image_classification/capi)
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir
make
```
# Run
If everything built successfully, you can use a script like below:
```
#!/bin/bash
./infer_image_classification \
        --infer_model=<path_to_directory_with_model> \
        --data_list=<path_to>/ILSVRC2012/val_list.txt \
        --data_dir=<path_to>/ILSVRC2012/ \
        --batch_size=50 \
        --skip_batch_num=10 \
        --iterations=500  \
        --profile \
        --use_mkldnn \
        --paddle_num_threads=<num_cpu_cores>
```
The command as above requires the inference model (passed via the `infer_model`
option) to return accuracy as the second output and model parameters to be
stored in separate files.

To run inference on a model without accuracy layer, with parameters stored
in a single file, with input image size 318x318, and without running passes, run:
```
#!/bin/bash
./infer_image_classification \
        --infer_model=<path_to_directory_with_model> \
        --data_list=<path_to>/ILSVRC2012/val_list.txt \
        --data_dir=<path_to>/ILSVRC2012/ \
        --batch_size=50 \
        --skip_batch_num=10 \
        --iterations=500  \
        --profile \
        --use_mkldnn \
        --paddle_num_threads=<num_cpu_cores>
        --with_labels=0 \
        --one_file_params=1 \
        --resize_size=318 \
        --crop_size=318 \
	--skip_passes
```
