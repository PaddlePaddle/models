# Linux GPU/CPU PYTHON 动转静精度测试

动转静精度测试的主程序为`test_export_shell.sh`，可以测试模型动转静的功能和inferecn预测结果的正确性。

<a name="1"></a>
## 1. 测试流程

该文档主要介绍 动转静精度验证 开发过程，将该脚本拷贝到相关repo中，直接运行 `test_export_shell.sh` 即可。

运行该脚本，会完成以下3个步骤：

1. 首先在模型库路径下创建 check_inference.py 文件，用于验证动态图预测精度和paddle.inference推理精度。
2. 改写模型库中的 export_model.py， 完成动转静操作后把模型路径、输入数据shape、模型结构作为参数传给 check_inference.py。
   ** 注意 ** 不同套件中这部分名称不同，可能需要根据套件代码进行修改。

3. 给定配置文件路径，遍历所有文件测试动转静功能。并把测试结果保存在 check_xx.log 文件中。（默认比对阈值为 1e-4）
