set -e

if [ ! -d pybind11 ]; then
    git clone https://github.com/pybind/pybind11.git
fi 

if [ ! -d kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi.git
fi 

python setup.py build_ext -i 
