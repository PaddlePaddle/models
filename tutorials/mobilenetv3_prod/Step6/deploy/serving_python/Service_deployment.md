
# 目录

- [1. 简介](#1---)
- [2. Paddle Serving服务化部署](#2---)
    - [2.1 准备测试数据](#2.1)
    - [2.2 准备环境](#2.2)
    - [2.3 准备inference模型](#2.3)
    - [2.4 准备服务化部署模型](#2.4)
    - [2.5 复制部署样例程序](#2.5)
    - [2.6 服务端修改](#2.6)
    - [2.7 客户端修改](#2.7)
    - [2.8 启动服务端模型预测服务 & 启动客服端](#2.8)
- [3. FAQ](#3)

## 1. 简介

Paddle Serving是飞桨开源的**服务化部署**框架，提供了C++ Serving和Python Pipeline两套框架，C++ Serving框架更倾向于追求极致性能，Python Pipeline框架倾向于二次开发的便捷性。旨在帮助深度学习开发者和企业提供高性能、灵活易用的工业级在线推理服务，助力人工智能落地应用。


更多关于Paddle Serving的介绍，可以参考[Paddle Serving官网repo](https://github.com/PaddlePaddle/Serving)。

本文档主要介绍利用Paddle Serving框架实现飞桨模型（以MobilenetV3为例）的服务化部署。

## 2. Paddle Serving服务化部署
Paddle Serving服务化部署主要包括以下步骤：
<img width="729" alt="图片" src="https://user-images.githubusercontent.com/54695910/147932862-63f6804b-9030-4b5b-8901-59097c33a0ec.png">

其中设置了2个其中设置了2个核验点，分别为：
* 启动服务端
* 启动客户端

### 2.1 准备测试数据
从验证集或者测试集中抽出至少一张图像，用于后续推理过程验证，同时长传对应数据集的标签。
以上传daisy.jpg图像和[数据集对应的标签文件](imagenet.label)到 ‘community/repo_template/images/’路径。

### 2.2 准备环境

**docker**是一个开源的应用容器引擎，可以让应用程序更加方便地被打包和移植。Paddle Serving容器化部署建议在docker中进行Serving服务化部署。以下教程均以docker环境展开说明。

（1）以下安装docker的Paddle Serving环境，CPU/GPU版本二选一即可。

    1）docker环境安装（CPU版本）
  
    
  ```
  # 拉取并进入 Paddle Serving的 CPU Docker
  docker pull paddlepaddle/serving:0.7.0-devel
  docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-devel bash
  docker exec -it test bash
  ```
  
  
    2)docker环境安装（GPU版本）
    
  ```
  # 拉取并进入 Paddle Serving的GPU Docker
  docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
  nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
  nvidia-docker exec -it test bash**
  ```
  
（2）安装Paddle Serving三个安装包，分别是：paddle-serving-server，paddle-serving-client 和 paddle-serving-app。

  ```
  wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
  pip3 install paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
  wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.7.0-cp37-none-any.whl
  pip3 install paddle_serving_client-0.7.0-cp37-none-any.whl
  wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.7.0-py3-none-any.whl
  pip3 install paddle_serving_app-0.7.0-py3-none-any.whl
  ```

  Paddle Serving Server更多不同运行环境的whl包下载地址，请参考：[下载页面](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md)

(3)为docker环境下载工程

```
git clone https://github.com/PaddlePaddle/models.git
cd models/tutorials/mobilenetv3_prod/Step6
```
**【注意】**：为了便于管理，后续Paddle Serving部署文件都会保存到`models/tutorials/mobilenetv3_prod/Step6/deploy/Serving_python/`路径下。

### 2.3 准备inference模型 (根据模型情况确认是否跳过该步骤)

由于MobileNetV3暂时只提供了预训练模型，因此需要先转换为Inference模型。若后期提供，可省略该步骤。在tools文件夹下提供了输出inference模型的脚本文件export_model.py，运行如下命令即可获取inference模型。

```
python3 ./tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams  --save-inference-dir=./deploy/Serving_python/mobilenetv3_model
```
在mobilenetv3_model文件夹下有inference.pdmodel、inference.pdiparams和inference.pdiparams.info文件。

### 2.4 准备服务化部署模型

【基本流程】

为了便于模型服务化部署，需要将静态图模型(模型结构文件：\*.pdmodel和模型参数文件：\*.pdiparams)使用paddle_serving_client.convert按如下命令转换为服务化部署模型：

```
python3 -m paddle_serving_client.convert --dirname {静态图模型路径} --model_filename {模型结构文件} --params_filename {模型参数文件} --serving_server {转换后的服务器端模型和配置文件存储路径} --serving_client {转换后的客户端模型和配置文件存储路径}
```
上面命令中 "转换后的服务器端模型和配置文件" 将用于后续服务化部署。其中`paddle_serving_client.convert`命令是`paddle_serving_client` whl包内置的转换函数，无需修改。

【实战】

针对MobileNetV3网络，将inference模型转换为服务化部署模型的示例命令如下，转换完后在本地生成**serving_server**和**serving_client**两个文件夹。本教程后续主要使用serving_server文件夹中的模型。

```
cd deploy/Serving_python/
python3 -m paddle_serving_client.convert \
    --dirname ./mobilenetv3_model/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --serving_server serving_server \
    --serving_client serving_client
```
【注意】：0.7.0版本的 PaddleServing 需要和PaddlePaddle 2.2之后的版本搭配进行模型转换.如果最开始使用docker安装整个环境，则Paddle Serving和PaddlePaddle版本已做好对应。   

### 2.5 复制部署样例程序
**【基本流程】**

服务化部署的样例程序的目录地址为：`**/models/docs/tipc/serving/template/code`

该目录下面包含3个文件，具体如下：

- web_service.py：用于开发**服务端模型预测**相关程序。由于使用多卡或者多机部署预测服务，设计高效的服务调度策略比较复杂，Paddle Serving将网络预测进行了封装，在这个程序里面开发者只需要关心部署服务引擎的初始化，模型预测的前处理和后处理开发，不用关心模型预测调度问题。

- config.yml：服务端模型预测相关**配置文件**，里面有各参数的详细说明。开发者只需要关注如下配置：http_port（服务的http端口），model_config（服务化部署模型的路径），device_type（计算硬件类型），devices（计算硬件ID）。

- pipeline_http_client.py：用于**客户端**访问服务的程序，开发者需要设置url（服务地址）、logid（日志ID）和测试图像。

**【实战】**

如果服务化部署MobileNetV3网络，需要拷贝上述三个文件以及上述导出的serving_server、serving_client文件夹到运行目录，建议在`**/models/community/repo_template/deploy/pdserving/`目录下。
```
cp -r **/models/community/repo_template/deploy/pdserving/*  ./

```
### 2.6 服务端修改

服务端修改包括服务端代码修改（即对web_service.py代码进行修改）和服务端配置文件（即对config.yml代码进行修改）修改。
服务端代码修改主要修改：初始化部署引擎、开发数据预处理程序、开发预测结果后处理程序三个模块。
服务端配置文件主要修改：
【注意】后续代码修改，主要修改包含`TIPC`字段的代码模块。

#### 2.6.1 初始化服务端部署配置引擎
**【基本流程】**

针对模型名称，修改web_service.py中类TIPCExampleService、TIPCExampleOp的名称，以及这些类初始化中任务名称name。


**【实战】**

针对MobileNetV3网络

（1）修改web_service.py文件后的代码如下：

```
from paddle_serving_server.web_service import WebService, Op
class MobileNetOp(Op):
    def init_op(self):
        pass
    def preprocess(self, input_dicts, data_id, log_id):
        pass
    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        pass
        
class MobileNetv3Service(WebService):
    def get_pipeline_response(self, read_op):
        mobilenet_op = MobileNetOp(name="imagenet", input_ops=[read_op])
        return mobilenet_op
uci_service = MobileNetv3Service(name="imagenet")
uci_service.prepare_pipeline_config("config.yml")
uci_service.run_service()
```

#### 2.6.2 开发数据预处理程序

**【基本流程】**

web_service.py文件中的TIPCExampleOp类的preprocess函数用于开发数据预处理程序，包含输入、处理流程和输出三部分。

**（1）输入：** 一般开发者使用时，只需要关心input_dicts和log_id两个输入参数。这两个参数与客户端访问服务程序tipc_pipeline_http_client.py中的请求参数相对应，即：
```
    data = {"key": ["image"], "value": [image], "logid":logid}
```
其中key和value的数据类型都是列表，并且一一对应。input_dicts是一个字典，它的key和value和data中的key和value是一致的。log_id和data中的logid一致。

**（2）处理流程：** 数据预处理流程和基于Paddle Inference的模型预处理流程相同。

**（3）输出：** 需要返回四个参数，一般开发者使用时只关心第一个返回值，网络输入字典，其余返回值使用默认的即可。

```
{"input": input_imgs}, False, None, ""
```
上述网络输入字典的key可以通过服务化模型配置文件serving_server/serving_server_conf.prototxt中的feed_var字典的name字段获取。

**【实战】**

针对MobileNetV3网络的数据预处理开发，修改web_service.py文件中代码如下：

添加头文件：

```py
import sys
import logging
import numpy as np
import base64, cv2
from paddle_serving_app.reader import Sequential, URL2Image, Resize, CenterCrop, RGB2BGR, Transpose, Div, Normalize, Base64ToImage
```     
修改MobileNetOp中的init_op和preprocess函数相关代码：

```py
class MobileNetOp(Op):
    def init_op(self):
        self.seq = Sequential([
            Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                True)
        ])
        self.label_dict = {}
        label_idx = 0
        with open("../../images/imagenet.label") as fin:
            for line in fin:
                self.label_dict[label_idx] = line.strip()
                label_idx += 1
    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = self.seq(im)
            imgs.append(img[np.newaxis, :].copy())
        input_imgs = np.concatenate(imgs, axis=0)
        return {"input": input_imgs}, False, None, ""
```

#### 2.6.3 开发预测结果后处理程序

【基本流程】

web_service.py文件中的TIPCExampleOp类的 postprocess 函数用于开发预测结果后处理程序，包含输入、处理流程和输出三部分。

**（1）输入：** 包含四个参数，其中参数input_dicts、log_id和数据预处理函数preprocess中一样，data_id可忽略，fetch_dict 是网络预测输出字典，其中输出的key可以通过服务化模型配置文件serving_server/serving_server_conf.prototxt中的fetch_var字典的name字段获取。

**（2）处理流程：** 数据预处理流程和基于Paddle Inference的预测结果后处理一致。

**（3）输出：** 需要返回三个参数，一般开发者使用时只关心第一个返回值，预测结果字典，其余返回值使用默认的即可。

```
result, None, ""
```

【实战】

针对MobileNet网络的预测结果后处理开发，修改web_service.py文件中MobileNetOp中的postprocess函数相关代码如下：

```py
    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        score_list = fetch_dict["softmax_1.tmp_0"]
        result = {"label": [], "prob": []}
        for score in score_list:
            score = score.tolist()
            max_score = max(score)
            result["label"].append(self.label_dict[score.index(max_score)]
                                   .strip().replace(",", ""))
            result["prob"].append(max_score)
        result["label"] = str(result["label"])
        result["prob"] = str(result["prob"])
        return result, None, ""
```
#### 2.6.4 修改服务配置文件config.yml

- http_port：使用默认的端口号18080
- OP名称：第14行修改成imagenet； （实际自己项目中，与2.6.1中name设置保持一致）
- model_config：与2.4转换后服务化部署模型文件夹路径一致，这里使用默认配置 "./serving_server"
- device_type：使用默认配置1，基于GPU预测；使用参数0，基于CPU预测。
- devices：使用默认配置"0"，0号卡预测     

### 2.7 客户端修改

修改pipeline_http_client.py程序，用于访问2.6中的服务端服务。

**【基本流程】**
主要设置客户端代码中的url（服务地址）、logid（日志ID）和测试图像。其中服务地址的url的样式为 "http://127.0.0.1:18080/tipc_example/prediction" ，url的设置需要将url中的tipc_example更新为TIPCExampleService类初始化的name。

**【实战】**

针对MobileNet网络, 修改pipeline_http_client.py程序中的url（服务地址）、logid（日志ID）和测试图像地址，其中url改为：

```
url = "http://127.0.0.1:18080/imagenet/prediction"
``` 

### 2.8 启动服务端模型预测服务 & 启动客服端

### 2.8.1 启动服务端模型预测服务
**【基本流程】**

当完成服务化部署引擎初始化、数据预处理和预测结果后处理开发，则可以按如下命令启动模型预测服务：

```bash
python3 web_service.py &
```                               
**【实战】**

针对MobileNet网络, 启动成功的界面如下：

![图片](https://user-images.githubusercontent.com/54695910/147933042-13279f61-b5ba-4b4a-8841-8aaa29ec2bfa.png)
   

#### 2.8.2 启动客户端，访问服务

**【基本流程】**

当成功启动了模型预测服务，可以启动服务端代码，用于访问2.8.1中的服务端服务。

**【实战】**

针对MobileNet网络, 启动成功的界面如下：
       
客户端访问服务的命令如下：

```
python3 pipeline_http_client.py
```                                                  
访问成功的界面如下图：

![图片](https://user-images.githubusercontent.com/54695910/147933077-e07e19fc-0884-4341-a1f3-4d090c1e9608.png)


【注意事项】
如果访问不成功，可能设置了代理影响的，可以用下面命令取消代理设置。

```
unset http_proxy
unset https_proxy
```



## 3. FAQ

如果您在使用该文档完成Paddle Serving服务化部署的过程中遇到问题，可以给在[这里](https://github.com/PaddlePaddle/Serving/issues)提一个ISSUE，我们会高优跟进。
