#!/bin/bash

if [ -d ./model_baseline/ ]
then
    echo "./model_baseline/ directory already existed, exiting"
    exit -1
fi

if [ -d ./model_finetuned/ ]
then
    echo "./model_finetuned/ directory already existed, exiting"
    exit -1
fi

if [ -d ./data/ ]
then
    echo "./data/ directory already existed, exiting"
    exit -1
fi

# download baseline model file to ./model_baseline/
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-1.0.0.tar.gz
tar xvf lexical_analysis-1.0.0.tar.gz
/bin/rm lexical_analysis-1.0.0.tar.gz

# download finetuned model file to ./model_finetuned/
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis_finetuned-1.0.0.tar.gz
tar xvf lexical_analysis_finetuned-1.0.0.tar.gz
/bin/rm lexical_analysis_finetuned-1.0.0.tar.gz

# download dataset file to ./data/
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-dataset-1.0.0.tar.gz
tar xvf lexical_analysis-dataset-1.0.0.tar.gz
/bin/rm lexical_analysis-dataset-1.0.0.tar.gz
