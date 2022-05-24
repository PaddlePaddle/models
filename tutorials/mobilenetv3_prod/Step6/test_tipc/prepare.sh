#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# MODE be one of ['lite_train_lite_infer'，'lite_train_whole_infer' 
#                  'whole_train_whole_infer', 'whole_infer']          

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[12]}")

pip install -r requirements.txt
if [ ${MODE} = "lite_train_lite_infer" ];then
    # prepare lite data
    tar -xf ./test_images/lite_data.tar
    ln -s ./lite_data/ ./data
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    # prepare whole data
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi
    
elif [ ${MODE} = "lite_train_whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "serving_infer" ];then
    # get data
    tar -xf ./test_images/lite_data.tar
    # wget model
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./inference https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar  --no-check-certificate
        cd ./inference && tar xf mobilenet_v3_small_infer.tar && cd ../
    fi
    unset https_proxy
    unset http_proxy
elif [ ${MODE} = "cpp_infer" ];then
    PADDLEInfer=$3
    # wget model
    wget -nc -P  ./deploy/inference_cpp/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar  --no-check-certificate
    cd ./deploy/inference_cpp/ && tar xf mobilenet_v3_small_infer.tar
    if [ "" = "$PADDLEInfer" ];then
        wget -nc https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz --no-check-certificate
    else
        wget -nc $PADDLEInfer --no-check-certificate
    fi
    tar zxf paddle_inference.tgz
    if [ ! -d "paddle_inference" ]; then
        ln -s paddle_inference_install_dir paddle_inference
    fi
    wget -nc https://paddleocr.bj.bcebos.com/libs/opencv/opencv-3.4.7.tar.gz --no-check-certificate
    tar zxf opencv-3.4.7.tar.gz
    # build opencv
    cd opencv-3.4.7/
    root_path=$PWD
    install_path=${root_path}/opencv3
    build_dir=${root_path}/build

    rm -rf ${build_dir}
    mkdir ${build_dir}
    cd ${build_dir}

    cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON
    make -j
    make install
    cd ../../
    # build cpp
    bash tools/build.sh

elif [ ${MODE} = "paddle2onnx_infer" ];then
    # install paddle2onnx
    python_name_list=$(func_parser_value "${lines[2]}")
    IFS='|'
    array=(${python_name_list})
    python_name=${array[0]}
    ${python_name} -m pip install paddle2onnx
    ${python_name} -m pip install onnxruntime==1.9.0
    # get data
    tar -xf ./test_images/lite_data.tar
    # get model
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./inference https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar  --no-check-certificate
        cd ./inference && tar xf mobilenet_v3_small_infer.tar && cd ../
    fi
fi
