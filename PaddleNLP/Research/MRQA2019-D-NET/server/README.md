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
```
bash start.sh
```
