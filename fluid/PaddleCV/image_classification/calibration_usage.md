# Introduction of Accuracy Calibration Tool for 8 Bit Inference
The 8-bits (**INT8**) inference is also known as the Low Precision Inference which could speed up the inference with the lower accuracy loss. It has higher throughput and lower memory requirements compared to FP32. As the PaddlePaddle enables the INT8 inference supporting, we release a accuracy tool(Calibration.py) at the same time. This tool will generate the  quantization parameters and quantized model file finally.


## Usage
1. Build the PaddlePaddle with MKLDNN supporting.
2. `cd /path/to/the/model/fluid/PaddleCV/image_classification`
3. `export FLAGS_use_mkldnn=True`
4. Run the command `python calibration.py --model=MobileNet  --batch_size=50  --class_dim=1000  --image_shape=3,224,224  --with_mem_opt=True  --use_gpu=False  --pretrained_model=weights/mobilenet --out=quantized_out --algo=KL`
    It will generate the specified model (by the parameter --model) to the output directory "quantized_out". Tht tool also provide the parameter "--algo" for KL divergence algorithm which will improve the accuracy.
5. Run the INT8 inference with this command `python eval_int8.py  --iterations=1000  --batch_size=50  --class_dim=1000  --image_shape=3,224,224  --with_mem_opt=True  --use_gpu=False --pretrained_model=quantized_out`. 

## Result
1. Accuracy

| Topology | FP32 Accuracy(Top-1/Top-5) | INT8 Accurary(Top-1/Top-5) |
| --- | :---: | :---: |
ResNet-50 (FB) | 76.63%/93.10%| 76.42%/93.07% 
MobileNet-V1 | 70.78%/89.69% | 70.10%/89.30%

2. Performance

| Topology | FP32 Throughput  | INT8 Throughput | FP32 latency(1x1)  | INT8 latency(1x1) |
| --- | :---: | :---: | :---: | :---: |
ResNet-50 (FB) | 260 | 527 | 62.4 |36.7 
MobileNet-V1 | 1207 |1934 | 9.3 | 8.4

Note: The above performance measured on SKX8180 1S (HT On, Turbo On) 
(Throughput imgs/sec; Latency: ms; 1x1: batch size 1 x thread 1)
