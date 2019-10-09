# server

## Introduction 
MRQA 2019 Shared Task submission will be handled through the [Codalab](https://worksheets.codalab.org/) platform: see [these instructions](https://worksheets.codalab.org/worksheets/0x926e37ac8b4941f793bf9b9758cc01be/).

We provided D-NET models submission environment for MRQA competition. it includes two server: bert server and xlnet server, we merged the results of two serves.

## Inference Model Preparation 
Download bert inference model and xlnet inferece model
```
bash wget_server_inference_model.sh
```

## Start server

We can set GPU card for bert server or xlnet server, By setting variable CUDA_VISIBLE_DEVICES:
```
export CUDA_VISIBLE_DEVICES=1
```
In main_server.py file we set the server port for bert and xlnet model, as shown below, If the port 5118 or 5120 is occupied, please set up an idle port. 
```
url_1 = 'http://127.0.0.1:5118'   # url for model1
url_2 = 'http://127.0.0.1:5120'   # url for model2
```
start server
```
bash start.sh
```
