# 文本分类

以下是本例的简要目录结构及说明：

```text
.
|-- README.md               # README
|-- data_generator          # IMDB数据集生成工具
|   |-- IMDB.py             # 在data_generator.py基础上扩展IMDB数据集处理逻辑
|   |-- build_raw_data.py   # IMDB数据预处理，其产出被splitfile.py读取。格式：word word ... | label
|   |-- data_generator.py   # 与AsyncExecutor配套的数据生成工具框架
|   `-- splitfile.py        # 将build_raw_data.py生成的文件切分，其产出被IMDB.py读取
|-- data_generator.sh       # IMDB数据集生成工具入口
|-- data_reader.py          # 预测脚本使用的数据读取工具
|-- infer.py                # 预测脚本
`-- train.py                # 训练脚本
```

## 简介

本目录包含用fluid.AsyncExecutor训练文本分类任务的脚本。网络模型定义沿用自父目录nets.py

## 训练

1. 运行命令 `sh data_generator.sh`，下载IMDB数据集，并转化成适合AsyncExecutor读取的训练数据
2. 运行命令 `python train.py bow` 开始训练模型。
    ```python
    python train.py bow    # bow指定网络结构，可替换成cnn, lstm, gru
    ```

3. (可选）想自定义网络结构，需在[nets.py](../nets.py)中自行添加，并设置[train.py](./train.py)中的相应参数。
    ```python
    def train(train_reader,     # 训练数据
        word_dict,              # 数据字典
        network,                # 模型配置
        use_cuda,               # 是否用GPU
        parallel,               # 是否并行
        save_dirname,           # 保存模型路径
        lr=0.2,                 # 学习率大小
        batch_size=128,         # 每个batch的样本数
        pass_num=30):           # 训练的轮数
    ```

## 训练结果示例

```text
pass_id: 0 pass_time_cost 4.723438
pass_id: 1 pass_time_cost 3.867186
pass_id: 2 pass_time_cost 4.490111
pass_id: 3 pass_time_cost 4.573296
pass_id: 4 pass_time_cost 4.180547
pass_id: 5 pass_time_cost 4.214476
pass_id: 6 pass_time_cost 4.520387
pass_id: 7 pass_time_cost 4.149485
pass_id: 8 pass_time_cost 3.821354
pass_id: 9 pass_time_cost 5.136178
pass_id: 10 pass_time_cost 4.137318
pass_id: 11 pass_time_cost 3.943429
pass_id: 12 pass_time_cost 3.766478
pass_id: 13 pass_time_cost 4.235983
pass_id: 14 pass_time_cost 4.796462
pass_id: 15 pass_time_cost 4.668116
pass_id: 16 pass_time_cost 4.373798
pass_id: 17 pass_time_cost 4.298131
pass_id: 18 pass_time_cost 4.260021
pass_id: 19 pass_time_cost 4.244411
pass_id: 20 pass_time_cost 3.705138
pass_id: 21 pass_time_cost 3.728070
pass_id: 22 pass_time_cost 3.817919
pass_id: 23 pass_time_cost 4.698598
pass_id: 24 pass_time_cost 4.859262
pass_id: 25 pass_time_cost 5.725732
pass_id: 26 pass_time_cost 5.102599
pass_id: 27 pass_time_cost 3.876582
pass_id: 28 pass_time_cost 4.762538
pass_id: 29 pass_time_cost 3.797759
```
与fluid.Executor不同，AsyncExecutor在每个pass结束不会将accuracy打印出来。为了观察训练过程，可以将fluid.AsyncExecutor.run()方法的Debug参数设为True，这样每个pass结束会把参数指定的fetch variable打印出来：

```
async_executor.run(
    main_program,
    dataset,
    filelist,
    thread_num,
    [acc],
    debug=True)
```

## 预测

1. 运行命令 `python infer.py bow_model`, 开始预测。
    ```python
    python infer.py bow_model     # bow_model指定需要导入的模型
    ```

## 预测结果示例
```text
model_path: bow_model/epoch0.model, avg_acc: 0.882600
model_path: bow_model/epoch1.model, avg_acc: 0.887920
model_path: bow_model/epoch2.model, avg_acc: 0.886920
model_path: bow_model/epoch3.model, avg_acc: 0.884720
model_path: bow_model/epoch4.model, avg_acc: 0.879760
model_path: bow_model/epoch5.model, avg_acc: 0.876920
model_path: bow_model/epoch6.model, avg_acc: 0.874160
model_path: bow_model/epoch7.model, avg_acc: 0.872000
model_path: bow_model/epoch8.model, avg_acc: 0.870360
model_path: bow_model/epoch9.model, avg_acc: 0.868480
model_path: bow_model/epoch10.model, avg_acc: 0.867240
model_path: bow_model/epoch11.model, avg_acc: 0.866200
model_path: bow_model/epoch12.model, avg_acc: 0.865560
model_path: bow_model/epoch13.model, avg_acc: 0.865160
model_path: bow_model/epoch14.model, avg_acc: 0.864480
model_path: bow_model/epoch15.model, avg_acc: 0.864240
model_path: bow_model/epoch16.model, avg_acc: 0.863800
model_path: bow_model/epoch17.model, avg_acc: 0.863520
model_path: bow_model/epoch18.model, avg_acc: 0.862760
model_path: bow_model/epoch19.model, avg_acc: 0.862680
model_path: bow_model/epoch20.model, avg_acc: 0.862240
model_path: bow_model/epoch21.model, avg_acc: 0.862280
model_path: bow_model/epoch22.model, avg_acc: 0.862080
model_path: bow_model/epoch23.model, avg_acc: 0.861560
model_path: bow_model/epoch24.model, avg_acc: 0.861280
model_path: bow_model/epoch25.model, avg_acc: 0.861160
model_path: bow_model/epoch26.model, avg_acc: 0.861080
model_path: bow_model/epoch27.model, avg_acc: 0.860920
model_path: bow_model/epoch28.model, avg_acc: 0.860800
model_path: bow_model/epoch29.model, avg_acc: 0.860760
```
注：过拟合导致acc持续下降，请忽略
