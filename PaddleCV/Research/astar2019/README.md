### 百度之星轻量化检测比赛评测工具

数据目录结构如下：

```
your/path/coco/
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
|   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
|   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
|   ...

```

命令示例: 
```bash
# Evaluate
python score.py --model_dir your/path/saved_model/ --data_dir your/path/coco/
```
