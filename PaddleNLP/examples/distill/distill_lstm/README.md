# distill lstm

## 训练bert finetuneing模型

```shell
export CUDA_VISIBLE_DEVICES=7
export TASK_NAME=SST-2

python -u ./run_bert_finetune.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir ./model/$TASK_NAME/ \
    --n_gpu 1 \

```
##  训练蒸馏模型
将bert的知识蒸馏到基于BiLSTM的小模型里

```shell
python bert_distill.py
```
