# 使用ParallelExecutor的中文命名实体识别示例

以下是本例的简要目录结构及说明：

```text
.
├── data                 # 存储运行本例所依赖的数据，从外部获取
├── reader.py            # 数据读取接口, 从外部获取
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
```

## 数据
在data目录下，有两个文件夹，train_files中保存的是训练数据，test_files中保存的是测试数据，作为示例，在目录下我们各放置了两个文件，实际训练时，根据自己的实际需要将数据放置在对应目录，并根据数据格式，修改reader.py中的数据读取函数。

## 训练
修改 [train.py](./train.py) 的 `main` 函数，指定数据路径，运行`python train.py`开始训练。

训练记录形如
```txt
pass_id:0, time_cost:4.92960214615s
[Train] precision:0.000862136531076, recall:0.0059880239521, f1:0.00150726226363
[Test] precision:0.000796178343949, recall:0.00335758254057, f1:0.00128713933283
pass_id:1, time_cost:0.715255975723s
[Train] precision:0.00474094141551, recall:0.00762112139358, f1:0.00584551148225
[Test] precision:0.0228873239437, recall:0.00727476217124, f1:0.0110403397028
pass_id:2, time_cost:0.740842103958s
[Train] precision:0.0120967741935, recall:0.00163309744148, f1:0.00287769784173
[Test] precision:0, recall:0.0, f1:0
```

## 预测
修改 [infer.py](./infer.py) 的 `infer` 函数，指定：需要测试的模型的路径、测试数据、预测标记文件的路径，运行`python infer.py`开始预测。

预测结果如下
```txt
152804  O       O
130048  O       O
38862   10-B    O
784     O       O
1540    O       O
4145    O       O
2255    O       O
0       O       O
1279    O       O
7793    O       O
373     O       O
1621    O       O
815     O       O
2       O       O
247     24-B    O
401     24-I    O
```
输出分为三列，以"\t"分割，第一列是输入的词语的序号，第二列是标准结果，第三列为标记结果。多条输入序列之间以空行分隔。
