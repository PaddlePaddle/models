#!/bin/bash

# download model file to ./model/
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lac-1.0.0.tar.gz
tar xvf lac-1.0.0.tar.gz
/bin/rm lac-1.0.0.tar.gz

# download dataset file to ./data/
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lac-dataset-1.0.0.tar.gz
tar xvf lac-dataset-1.0.0.tar.gz
/bin/rm lac-dataset-1.0.0.tar.gz
