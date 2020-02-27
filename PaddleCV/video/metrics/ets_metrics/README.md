## ActivityNet Captions 指标计算

- ActivityNet Captions的指标评估代码可以参考[官方网站](https://github.com/ranjaykrishna/densevid_eval)

- 下载指标评估代码，将coco-caption和evaluate.py拷贝到PaddleVideo下；

- 计算精度指标，python运行evaluate.py文件，可通过-s参数指定结果文件，-r参数修改标签文件；

- 由于模型计算波动较大，在评估过程中可以取不同Epoch的所得训练模型计算精度指标，最优的METEOR值约为10.0左右。
