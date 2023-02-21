## 1. 推理 Benchmark

### 1.1 软硬件环境
PP-Helixfold模型的推理测试是在NVIDIA A100 (40G)单卡上完成的，batch size大小为1。为了能复现我们论文中报告的实验结果，需在特定环境下进行实验。

* Python: 3.7
* CUDA 11.6
* CUDNN 8.4.0
* NCCL 2.14.13

### 1.2 数据集
PP-HelixFold模型使用的训练样本25%来自RCSB PDB，75%来自自蒸馏数据集。测试时，我们搜集了87个CASP14的结构域蛋白和60个从2022-08-01到2022-08-31的CAMEO蛋白作为测试集。

### 1.3 模型效果
通过与原版AlphaFold2模型和哥伦比亚大学Mohammed AlQuraishi教授团队基于PyTorch复现的OpenFold模型的性能对比测试显示，PP-HelixFold模型的训练性能相比AlphaFold2提升106.97%，相比OpenFold提升104.86%，将训练耗时从约11天减少到5.12天，并且在使用混合并行时能进一步降低至2.89天。在性能大幅度提升的同时，PP-HelixFold从头端到端完整训练可以达到AlphaFold2论文媲美的精度。在包含87个蛋白的CASP14数据集和60个蛋白的CAMEO数据集上，PP-HelixFold模型的TM-score指标分别达到0.8771和0.9026，与原版AlphaFold2准确率相当甚至更优。

![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_computational_perf.png)
![](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/.github/HelixFold_infer_accuracy.png)


## 2. 相关使用说明
请参考：https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold
