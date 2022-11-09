## 1. Inference Benchmark

### 1.1 Environment
The PP-Helixfold model's inference test is tested on single-card NVIDIA A100 (40G), batch size=1. To reproduce the results reported in our paper, specific environment settings are required as below.

* Python: 3.7
* CUDA 11.2
* CUDNN 8.10.1
* NCCL 2.12.12.

### 1.2 Datasets
For training, the PP-Helixfold model uses 25% of samples from RCSB PDB and 75% of self-distillation samples. For evaluation, we collect 87 domain targets from CASP14 and 371 protein targets from CAMEO, ranging from 2021-09-04 to 2022-02-19.

### 1.3 Performance
Compared with the computational performance of AlphaFold2 reported in the paper and OpenFold implemented through PyTorch, PP-Helixfold reduces the training time from about 11 days to 7.5 days, and it can be further reduced to only 5.3 days when using hybrid parallelism. Training PP-Helixfold from scratch can achieve competitive accuracy with AlphaFold2.

![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_computational_performance.png)
![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_accuracy.png)


## 2. Reference
Ref: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold
