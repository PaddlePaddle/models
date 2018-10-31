#!/bin/bash

wget https://pjreddie.com/media/files/darknet53.conv.74 -O darknet53.pretrain
echo "download finish"
python weight_parser.py pretrain
echo "parse finish"
