#!/bin/bash

# download pretrain model file to ./models/
MODEL_URL=https://baidu-nlp.bj.bcebos.com/emotion_detection_textcnn-1.0.0.tar.gz
wget --no-check-certificate ${MODEL_URL}

tar xvf emotion_detection_textcnn-1.0.0.tar.gz
/bin/rm emotion_detection_textcnn-1.0.0.tar.gz
