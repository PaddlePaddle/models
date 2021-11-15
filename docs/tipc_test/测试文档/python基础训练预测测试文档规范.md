# 背景

基础训练文档对应基础训练预测功能测试，主程序为`test_train_inference_python.sh`。本文介绍基础训练预测在Linux端，Mac端，Windows端，Jeston端的使用方法。

# 文档规范

本文档为TIPC的基础训练预测功能测试文档。包含linux端，Mac端，Windows端，jeston端的测试方法。

### 1.基础训练预测规范
基础训练预测工具测试的功能包括训练和Python预测两部分，汇总结论也分为`训练相关`和`预测相关`两部分。
下面以[PaddleOCR样板间](https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/test_tipc)为例说明目录结构规范。
```shell
test_tipc/
├── configs/  # 配置文件目录
	├── ppocr_det_mobile/                #  ppocr检测模型mobile版本的配置文件和参数文件夹
		├── det_mv3_db.yml               # 测试mobile版ppocr检测模型训练的yml文件
		├── train_infer_python.txt.      # 测试mobile版ppocr检测模型训练+预测的参数配置文件
		├── train_linux_cpu_normal_normal_infer_python_mac.txt     # Mac端测试mobile版ppocr检测模型的参数配置文件
		├── train_linux_gpu_normal_normal_infer_python_windows.txt     # Windows端测试mobile版ppocr检测模型的参数配置文件
		├── model_linux_gpu_normal_normal_infer_python_jetson.txt   # Jeston端测试mobile版ppocr检测模型的参数配置文件
		├── ...                                
├── results/   # 预先保存的预测结果，用于和实际预测结果进行精读比对
	├── python_ppocr_det_mobile_results_fp32.txt           # 预存的mobile版ppocr检测模型python预测fp32精度的结果
	├── python_ppocr_det_mobile_results_fp16.txt           # 预存的mobile版ppocr检测模型python预测fp16精度的结果
	├── cpp_ppocr_det_mobile_results_fp32.txt       # 预存的mobile版ppocr检测模型c++预测的fp32精度的结果
	├── cpp_ppocr_det_mobile_results_fp16.txt       # 预存的mobile版ppocr检测模型c++预测的fp16精度的结果
	├── ...
├── prepare.sh                        # 完成test_*.sh运行所需要的数据和模型下载
├── test_train_inference_python.sh    # 测试python训练预测的主程序
├── test_inference_cpp.sh             # 测试c++预测的主程序
├── test_serving.sh                   # 测试serving部署预测的主程序
├── test_lite.sh                      # 测试lite部署预测的主程序
├── compare_results.py                # 用于对比log中的预测结果与results中的预存结果精度误差是否在限定范围内
└── readme.md                         # 使用文档
```
主要关注：
1. 所有工具位于`test_tipc`目录下，`test_tipc`目录位于套件根目录下；
2. `configs`目录存放测试所需的所有配置文件；
2. `results`目录存放精度测试所需的gt文件；
3. `doc`目录存放readme.md以外的其他子文档；
4. `prepare.sh`用于准备测试所需的模型、数据等；
5. `test_*.sh`为测试主程序，按照功能分为多个文件，命名格式为`test_[功能]_[语言].sh`。


（1）训练相关

内容：给出套件所有模型的基础训练预测打通情况汇总信息，表格形式呈现，须包含`算法名称`、`模型名称`、`单机单卡`、`单机多卡`、`多机多卡`、`模型压缩（单机多卡）`这6个字段。
	
- 算法名称：该模型对应的算法，可以是算法简称；
- 模型名称：与套件提供模型的名称对应；
- 单机单卡：单机单卡训练的支持情况，包括`正常训练`和`混合精度`两种模式，支持哪种模式就填写哪种。
- 单机多卡：同上。
- 多机多卡：同上。
- 模型压缩（单机多卡）：单机多卡模式下，支持的模型压缩算法，分别给出`正常训练`、`混合精度`、`离线量化`三种情况的模型压缩算法支持情况。

示例：
| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  DB  | ch_ppocr_mobile_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练：FPGM裁剪、PACT量化 <br> 离线量化（无需训练） |
|  DB  | ch_ppocr_server_v2.0_det| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练：FPGM裁剪、PACT量化 <br> 离线量化（无需训练） |
| CRNN | ch_ppocr_mobile_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练：PACT量化 <br> 离线量化（无需训练） |
| CRNN | ch_ppocr_server_v2.0_rec| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练：PACT量化 <br> 离线量化（无需训练） |
|PP-OCR| ch_ppocr_mobile_v2.0| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|PP-OCR| ch_ppocr_server_v2.0| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |
|PP-OCRv2| ch_PP-OCRv2 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 | - |


（2）预测相关

内容：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，表格形式给出这两类模型对应的预测功能汇总情况，包含`模型类型`、`device`、`batchsize`、`tensorrt`、`mkldnn`、`cpu多线程`这6个字段。
	
- 算法名称：该模型对应的算法，可以是算法简称；
- device：CPU或GPU，其他字段取值对应本字段指定的环境；
- batchsize：一般包括1、6两种batchsize，根据实际支持情况填写。
- tensorrt：开启tensorrt支持的精度，包括`fp32`、`fp16`、`int8`三种，当device为CPU时，本字段填`-`。
- mkldnn：开启mkldnn支持的精度，包括`fp32`、`fp16`、`int8`三种，当device为GPU时，本字段填`-`。
- cpu多线程：支持时填`支持`，不支持时留空即可，当device为GPU时，本字段填`-`。

基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1/6 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1/6 | - | fp32/fp16 | 支持 |
| 量化模型 | GPU | 1/6 | int8 | - | - |
| 量化模型 | CPU | 1/6 | - | int8 | 支持 |




### 2.测试流程
#### 2.1 安装依赖
安装测试本功能所需的依赖，包括`autolog`工具和PaddleSlim。
[autolog](https://github.com/LDOUBLEV/AutoLog)是一个辅助工具，用户规范化打印inference的输出结果。
autolog安装方式如下，参考文档：https://github.com/LDOUBLEV/AutoLog
```
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```

PaddleSlim安装方式如下，参考文档https://github.com/PaddlePaddle/PaddleSlim：
```
pip3 install paddleslim
```


#### 2.2 Linux端功能测试
内容：分别给出5种模式下的具体测试命令。
参考文档：https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/docs/test_train_inference_python.md#22-%E5%8A%9F%E8%83%BD%E6%B5%8B%E8%AF%95
示例：

先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。


`test_train_inference_python.sh`包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'lite_train_lite_infer'
```  

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt  'lite_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ../test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'lite_train_whole_infer'
```  

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ../test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'whole_infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'whole_infer' '1'
```  

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'whole_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt 'whole_train_whole_infer'
```  

- 模式5：klquant_whole_infer，测试离线量化；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt  'klquant_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_infer_python.txt  'klquant_whole_infer'
```

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如'lite_train_lite_infer'模式下，会运行训练+inference的链条，因此，在`test_tipc/output`文件夹有以下文件：
```
test_tipc/output/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_0_autocast_null/  # GPU 0号卡上正常训练的训练日志和模型保存文件夹
|- pact_train_gpus_0_autocast_null/  # GPU 0号卡上量化训练的训练日志和模型保存文件夹
......
|- python_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log  # CPU上开启Mkldnn线程数设置为1，测试batch_size=1条件下的预测运行日志
|- python_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log # GPU上开启TensorRT，测试batch_size=1的半精度预测日志
......
```


#### 2.3. MAC端功能测试

Mac端不包含GPU，并且CPU不支持mkldnn预测，因此在Mac端无需测试tipc中关于GPU和mkldnn相关的部分。

为了方便Mac端功能测试，把需要测试的功能拆分为了新的参数文件`./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt`

除了修改配置文件之外，其余测试方法同2.1节的功能测试方法：

先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。

`test_train_inference_python.sh`包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
# 同linux端运行不同的是，Mac端测试使用新的配置文件mac_ppocr_det_mobile_params.txt，
# 配置文件中默认去掉了GPU和mkldnn相关的测试链条
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_lite_infer'
```  

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'lite_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt  'lite_train_whole_infer'
```  

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_infer' '1'
```  

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；（Mac端不建议运行此模式）
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'whole_train_whole_infer'
```  

- 模式5：klquant_whole_infer，测试离线量化；
```shell
bash test_tipc/prepare.sh ./test_tipc/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt 'klquant_whole_infer'
bash test_tipc/test_train_inference_python.sh test_tipc/configs/ppocr_det_mobile/train_mac_cpu_normal_normal_infer_python_mac_cpu.txt  'klquant_whole_infer'
```


#### 2.4 Windows端功能测试

windows端的功能测试同Linux端，为了方便区分Windows端功能测试，把需要测试的功能拆分为了新的参数文件`./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt`

除了修改配置文件之外，其余测试方法同2.1节的功能测试方法。

先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。

`test_train_inference_python.sh`包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'lite_train_lite_infer'
```  

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'lite_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'lite_train_whole_infer'
```  

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'whole_infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'whole_infer' '1'
```  

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt 'whole_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt 'whole_train_whole_infer'
```  

- 模式5：klquant_whole_infer，测试离线量化；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt 'klquant_whole_infer'
bash test_tipc/test_train_inference_python.sh test_tipc/configs/ppocr_det_mobile/train_windows_gpu_normal_normal_infer_python_windows_cpu_gpu.txt  'klquant_whole_infer'
```

#### 2.5 Jeston端功能测试

Jeston产品是Nvidia推出的开发者套件，用于部署AI模型。Jeston系列的产品有Jeston Nano, TX，NX等等。本节以PaddleOCR检测模型和JestonNX为例，介绍如何在Jeston上接入TIPC预测推理的测试。

Jeston的CPU性能远差于笔记本或者台式机，因此在Jeston上，只需要测试GPU上预测相关的链条即可，包括GPU预测，GPU+TensorRT(fp32)，GPU+TensorRT(fp16)预测。

Jeston上无需测试TIPC训练部分，仅需要测试预测推理部分即可，因此，仅需要测试。

`test_train_inference_python.sh`的whole_infer模式：

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/model_linux_gpu_normal_normal_infer_python_jetson.txt 'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ppocr_det_mobile/model_linux_gpu_normal_normal_infer_python_jetson.txt 'whole_infer'
```


#### 2.6 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取日志中的预测坐标；
- 从本地文件中提取保存好的坐标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

运行命令：
```shell
python3.7 test_tipc/compare_results.py --gt_file=./test_tipc/results/python_*.txt  --log_file=./test_tipc/output/python_*.log --atol=1e-3 --rtol=1e-3
```

参数介绍：  
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- log_file: 指向运行test_tipc/test_train_inference_python.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：文本框，预测文本，类别等等，同样支持python_infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

运行结果：
正常运行效果如下：
```
Assert allclose passed! The results of python_infer_cpu_usemkldnn_False_threads_1_batchsize_1.log and ./test_tipc/results/python_ppocr_det_mobile_results_fp32.txt are consistent!
```

出现不一致结果时的运行输出：
```
......
Traceback (most recent call last):
  File "test_tipc/compare_results.py", line 140, in <module>
    format(filename, gt_filename))
ValueError: The results of python_infer_cpu_usemkldnn_False_threads_1_batchsize_1.log and the results of ./test_tipc/results/python_ppocr_det_mobile_results_fp32.txt are inconsistent!
```





