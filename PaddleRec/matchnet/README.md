# matchnet模型

以下是本例的简要目录和结构说明

```text
model.py        #模型结构
reader.py       #数据读取
train.py        #训练脚本
util.py         #工具函数
```
注：本示例所用数据非真实数据。需PaddlePaddle 1.7及以上版本或适当的develop版本才可以正常运行。

## 训练
### 训练命令
```bash
python train.py
```

### 参数解析：
```bash
python train.py -h
optional arguments:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        Training file
  --valid_file VALID_FILE
                        Validation file
  --epochs EPOCHS       Number of epochs for training
  --lr LR               learning rate
  --model_output_dir MODEL_OUTPUT_DIR
                        Model output folder
  --user_slots USER_SLOTS
                        Number of query slots
  --title_slots TITLE_SLOTS
                        Number of title slots
  --batch_size BATCH_SIZE
                        Batch size for training
  --embedding_dim EMBEDDING_DIM
                        Default Dimension of Embedding
  --sparse_feature_dim SPARSE_FEATURE_DIM
                        Sparse feature hashing spacefor index processing
  --random_ratio RANDOM_RATIO
                        random ratio for negative samples.
  --enable_ce           If set, run the task with continuous evaluation logs.
```
