#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tools/train.py -c configs/cascade_rcnn_r50_fpn_1x.yml -o max_iters=1000 enable_ce=true FasterRCNNTrainFeed.shuffle=false | python _ce.py
python tools/train.py -c configs/faster_rcnn_r50_fpn_1x.yml -o max_iters=1000 enable_ce=true FasterRCNNTrainFeed.shuffle=false | python _ce.py
python tools/train.py -c configs/mask_rcnn_r50_fpn_1x.yml -o max_iters=1000 enable_ce=true MaskRCNNTrainFeed.shuffle=false | python _ce.py
python tools/train.py -c configs/yolov3_darknet.yml -o max_iters=500 enable_ce=true YoloTrainFeed.shuffle=false | python _ce.py



