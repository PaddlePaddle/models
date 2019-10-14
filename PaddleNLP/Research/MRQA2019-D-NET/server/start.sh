#!/bin/bash

gpu_id=0

# start ernie service
# usage: sh start.sh port gpu_id
cd ernie_server
nohup sh start.sh 5118 $gpu_id > ernie.log 2>&1 &
cd ..

# start xlnet service
cd xlnet_server
nohup sh start.sh 5119 $gpu_id > xlnet.log 2>&1 &
cd ..

# start bert service
cd bert_server
nohup sh start.sh 5120 $gpu_id > bert.log 2>&1 &
cd ..

sleep 3
# start main server
# usage: python main_server.py --model_name
# the model_name specifies the model to be used in the ensemble.
nohup python main_server.py --ernie --xlnet > main_server.log 2>&1 &
