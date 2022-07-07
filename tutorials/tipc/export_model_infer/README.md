# Linux GPU/CPU PYTHON 动转静精度测试

动转静精度测试的主程序为`test_export_shell.sh`，可以测试模型动转静的功能和inferecn预测结果的正确性。

<a name="1"></a>
## 1. 开发说明

该文档主要介绍 动转静精度验证 开发过程，将该脚本拷贝到相关repo中，完成适配后直接运行 `test_export_shell.sh` 即可。

运行该脚本，会完成以下3个步骤：

1. 首先在模型库路径下创建 check_inference.py 文件，用于验证动态图预测精度和paddle.inference推理精度。
2. 改写模型库中的 export_model.py， 完成动转静操作后把模型路径、输入数据shape、模型结构作为参数传给 check_inference.py。
   ** 注意 ** 不同套件中这部分名称不同，可能需要根据套件代码进行修改。

3. 在 train_infer 链条中，自动升级 export_model 命令，检查推理一致性。（默认比对阈值为 1e-4）

<details>
<summary><b> 脚本实现细节（点击以展开详细内容或者折叠）</b></summary>
 
1. 创建 check_inference.py 文件
   将模版内容写入脚本，用于在启动模型动转静时测试精度误差
  
  ```
  def verify_paddle_inference_correctness(layer, path):
    from paddle import inference
    import numpy as np
    model_file_path = path + ".pdmodel"
    params_file_path = path + ".pdiparams"
    config = inference.Config(model_file_path, params_file_path)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_data = get_input_shape("xxxxxx")
    dygraph_input = {}
    if input_names == ["im_shape", "image", "scale_factor"]:
        input_names = ["image", "im_shape", "scale_factor"]
    for i,name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        fake_input = input_data[i]
        input_tensor.copy_from_cpu(fake_input)
        dygraph_input[name] = paddle.to_tensor(fake_input)
    predictor.run()
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    prob_out = output_tensors[0].copy_to_cpu()
    layer.eval()
    pred = layer(dygraph_input)
    pred = list(pred.values())[0] if isinstance(pred, dict) else pred
    correct = np.allclose(pred, prob_out, rtol=1e-4, atol=1e-4)
    absolute_diff = np.abs(pred.numpy() - prob_out)
    max_absolute_diff = np.max(absolute_diff)
    # print("max_absolute_diff:", max_absolute_diff)
    assert correct, "Result diff when load and inference:\nlayer max_absolute_diff:{}"\
                  .format(max_absolute_diff)
    print("Successful, dygraph and inference predictions are consistent.")'
  ```
    
2. 找到动转静执行文件

在 133-175 行中找到对应套件，修改
- 模型保存操作的真实路径--export_file
- 模型变量名--layer
- 模型保存路径--path

以 PaddleClas repo 为例，模型导出命令为 `python3.7 tools/export_model.py -c xxx`, 以tools/export_model.py 为入口，找到真正执行`jit.save`操作的文件： `ppcls/engine/engine.py`，具体代码在410行：

```
model = paddle.jit.to_static(
    model,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None] + self.config["Global"]["image_shape"],
            dtype='float32')
    ])
paddle.jit.save(model, save_path)
```

paddle.jit.save命令，保存模型为 model， 存储路径为 save_path， 预测输入shape为 [None] + self.config["Global"]["image_shape"], 由此可在 `test_export_shell.sh` 中为以下变量赋值：

```
    export_file=${root_path}/ppcls/engine/engine.py
    # define layer path and img_shape
    layer="model"
    path="save_path"
```

执行 `test_export_shell.sh` 脚本后，`ppcls/engine/engine.py` 中会添加5行代码，在每次执行动转静操作时自动测试精度：

```
model = paddle.jit.to_static(
    model,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None] + self.config["Global"]["image_shape"],
            dtype='float32')
    ])
paddle.jit.save(model, save_path)
# 以下为自动添加的代码
from check_inference import verify_paddle_inference_correctness
layer = model
path = save_path
verify_paddle_inference_correctness(layer, path)
```
  
</details>
  
  
  
## 2. 测试说明

测试需完成两项工作，下面以 PaddleClas repo 为例， 将 test_export_shell.sh 脚本拷贝到 PaddleClas 根目录下：

1. 根据config生成 check_inference.py 文件，以MobileNetV3为例:

```
bash test_export_shell.sh test_tipc/config/MobileNetV3/MobileNetV3_large_x0_5_train_infer_python.txt
```

2. 正常执行tipc “lite_train_lite_infer” 链条
```
bash test_tipc/test_train_inference_python.sh test_tipc/config/MobileNetV3/MobileNetV3_large_x0_5_train_infer_python.txt "lite_train_lite_infer"
```

完成之后结果会保存在 test_tipc/output/{model_name}/results_python.log 文件中：

```
Run successfully with command - python3.7 tools/export_model.py -c xxxxxx

```
