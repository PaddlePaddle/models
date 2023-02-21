## 1. Inference Benchmark

### 1.1 Environment
The PP-Helixfold model's inference test is tested on single-card NVIDIA A100 (40G), batch size=1. To reproduce the results reported in our paper, specific environment settings are required as below.

* Python: 3.7
* CUDA 11.6
* CUDNN 8.4.0
* NCCL 2.14.3

### 1.2 Datasets
For training, the PP-Helixfold model uses 25% of samples from RCSB PDB and 75% of self-distillation samples. For evaluation, we collect 87 domain targets from CASP14 and 60 protein targets from CAMEO, ranging from 2022-08-01 to 2022-08-31.

### 1.3 Performance
Compared with the computational performance of AlphaFold2 reported in the paper and OpenFold implemented through PyTorch, PP-Helixfold reduces the training time from about 11 days to 5.12 days, and it can be further reduced to only 2.89 days when using hybrid parallelism. Training PP-Helixfold from scratch can achieve competitive accuracy with AlphaFold2.

![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_computational_perf.png)
![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_infer_accuracy.png)


## 2. Reference
Ref: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold
