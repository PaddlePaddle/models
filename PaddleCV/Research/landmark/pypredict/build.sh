set -x
#http://www.paddlepaddle.org/documentation/docs/en/1.4/advanced_usage/deploy/inference/build_and_install_lib_en.html
alias wget='wget --no-check-certificat '
alias git="/usr/bin/git"

#set python home
export PYTHONHOME=~/.jumbo

wget https://paddle-inference-lib.bj.bcebos.com/1.4.1-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz
tar -xzf fluid_inference.tgz
mkdir so
cp `find fluid_inference -name '*.so*'`  so/

git clone https://github.com/pybind/pybind11.git
cd pybind11 && git checkout v2.2.4 && cd -

make

mv PyCNNPredict.so so
