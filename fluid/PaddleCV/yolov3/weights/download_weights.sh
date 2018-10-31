#!/bin/bash

wget https://pjreddie.com/media/files/yolov3.weights
echo "download finish"
python weight_parser.py yolov3
echo "parse finish"
