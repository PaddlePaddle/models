#!/bin/bash

# download dataset file to ./data/
if [ -d ./data/ ]
then
    echo "./data/ directory already existed, ignore download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-dataset-2.0.0.tar.gz
    tar xvf lexical_analysis-dataset-2.0.0.tar.gz
    /bin/rm lexical_analysis-dataset-2.0.0.tar.gz
fi

