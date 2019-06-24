# Accelerated Infer Project

This project is to accelerate the prediction of cnn.We need to compile theproject.

## Environment

Python2.7, Numpy

## Compile The Entire C++ Project

first open build.sh, and you need to set PYTHONHOME nev in build.sh
```
    export PYTHONHOME=/your/python/home
    sh build.sh
```

so folder will appear, This is the c++ program used to speed up the prediction.
then you can copy the so file to ../inference to predict models
```
    mv so ../inference
```
