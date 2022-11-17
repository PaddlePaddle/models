## 1. Training Benchmark

### 1.1 Environment

* The training process of PP-MSVSR model uses 8 GPUs, every GPU batch size is 2 for training. If the number GPU and batch size of training do not use the above configuration, you should refer to the FAQ to adjust the learning rate and number of iterations.

### 1.2 Datasets
The PP-MSVSR model uses REDS dataset for train and test. REDS consists of 240 training clips, 30 validation clips and 30 testing clips (each with 100 consecutive frames). Since the test ground truth is not available, we select four representative clips (they are '000', '011', '015', '020', with diverse scenes and motions) as our test set, denoted by REDS4. The remaining training and validation clips are re-grouped as our training dataset (a total of 266 clips).

### 1.3 Benchmark

| model | task | dataset | Parameter (M) |
|---|---|---|---|---|---|---|
|PP-MSVSR | Video Super-Resolution | REDS | 1.45 |

## 2. Inference Benchmark

### 2.1 Environment

* The PP-MSVSR model's inference test is tested with single-card V100, batch size=1, CUDA 10.2, CUDNN 7.5.1.

### 2.2 Datasets
The PP-MSVSR model uses REDS dataset for train and test. REDS consists of 240 training clips, 30 validation clips and 30 testing clips (each with 100 consecutive frames). Since the test ground truth is not available, we select four representative clips (they are '000', '011', '015', '020', with diverse scenes and motions) as our test set, denoted by REDS4. The remaining training and validation clips are re-grouped as our training dataset (a total of 266 clips).

### 2.3 Benchmark
| model | task | dataset | Parameter (M) | FLOPs (G) | PSNR | SSIM |
|---|---|---|---|---|---|---|
|PP-MSVSR | Video Super-Resolution | REDS4 | 1.45 | 111 | 31.2535 | 0.8884 |

## 3. Reference
Ref: https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
