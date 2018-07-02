set -e

if [ ! -d pybind11 ]; then
    git clone https://github.com/pybind/pybind11.git
fi 

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

python setup.py build_ext -i 
