# Linux GPU/CPU PYTHON 动转静精度测试

动转静精度测试的主程序为`test_export_shell.sh`，可以测试模型动转静的功能和inferecn预测结果的正确性。

<a name="1"></a>
## 1. 测试流程

该文档主要介绍 动转静精度验证 开发过程，将该脚本拷贝到相关repo中，完成适配后直接运行 `test_export_shell.sh` 即可。

运行该脚本，会完成以下3个步骤：

1. 首先在模型库路径下创建 check_inference.py 文件，用于验证动态图预测精度和paddle.inference推理精度。
2. 改写模型库中的 export_model.py， 完成动转静操作后把模型路径、输入数据shape、模型结构作为参数传给 check_inference.py。
   ** 注意 ** 不同套件中这部分名称不同，可能需要根据套件代码进行修改。

3. 给定配置文件路径，遍历所有文件测试动转静功能。并把测试结果保存在 check_xx.log 文件中。（默认比对阈值为 1e-4）


## 2. 适配流程

适配各套件需要完成以下几个步骤：

1. 找到动转静执行文件

在 72-104 行中找到对应套件，修改
- 模型保存操作的真实路径--export_file
- 模型变量名--layer
- 模型保存路径--path
- 数据尺寸--image_shape

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
    image_shape="self.config['Global']['image_shape']"
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
image_shape = self.config['Global']['image_shape']
verify_paddle_inference_correctness(layer, path, image_shape)
```


2. 遍历测试所有模型

run_all_config 函数用来遍历所有文件，并运行动转静脚本。需要根据不同套件改写 `export_model_cmd` 命令，并把配置文件所在的目录作为参数传给 run_all_config：

```
function run_all_config(){
for file in `ls $1`
do
    if [ -d $1"/"$file ]; then
        read_dir $1"/"$file
    else
        export_model_cmd="python3.7 tools/export_model.py -c $1'/'$file"
        status_log="./check_clas.log"
        eval $export_model_cmd
        last_status=${PIPESTATUS[0]}
        echo $last_status
        status_check $last_status "${file}" "${status_log}"
    fi
done
}

run_all_config "ppcls/configs/ImageNet/"
```
