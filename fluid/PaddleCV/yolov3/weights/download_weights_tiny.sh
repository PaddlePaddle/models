#! /usr/bin/env bash

wget https://pjreddie.com/media/files/yolov3-tiny.weights 
echo "download finish"
python weight_parser.py yolov3-tiny
echo "parse finish"
