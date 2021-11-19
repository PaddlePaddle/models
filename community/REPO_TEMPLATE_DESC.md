# REPO 提交规范

项目和代码的规范和可读性对于项目开发至关重要，可以提升开发效率。本文给出开发者在新建开发者生态项目时的repo目录示例，以供参考。

本文示例项目在文件夹[repo_template](./repo_template)下，您可以将这个文件夹中的内容拷贝出去，放在自己的项目文件夹下，并编写对应的代码与文档。

## 1. 目录结构

建议的目录结构如下：

```
./repo_template          # 项目文件夹名称，可以修改为自己的文件夹名称
|-- config               # 参数配置文件夹
|-- dataset              # 数据处理代码文件夹
|-- images               # 测试图片文件夹
|-- model                # 模型实现文件夹
|-- utils                # 功能类API文件夹
|-- deploy               # 预测部署相关
|   ├── pdinference      # 基于PaddleInference的python推理代码文件夹
|   ├── pdserving        # 基于PaddleServing的推理代码文件夹
|-- tools                # 工具类文件夹
|   ├── train.py         # 训练代码文件
|   ├── eval.py          # 评估代码文件
|   ├── infer.py         # 预测代码文件
|   ├── export.py         # 模型导出代码文件
|-- scripts              # 脚本类文件夹
|   ├── train.sh         # 训练脚本，需要包含单机单卡和单机多卡训练的方式，单机多卡的训练方式可以以注释的形式给出
|   ├── eval.sh          # 评估脚本，提供单机单卡的评估方式即可
|   ├── infer.sh         # 预测脚本
|   ├── export.sh        # 模型导出脚本
|-- test_tipc            # 训推一体测试文件夹
|-- README_en.md         # 英文用户手册
|-- README.md            # 中文用户手册
|-- LICENSE              # LICENSE文件
```

- **config：** 存储模型配置相关文件的文件夹，保存模型的配置信息，如 `configs.py、configs.yml` 等
- **dataset：** 存储数据相关文件的文件夹，包含数据下载、数据处理等，如 `dataset_download.py、dataset_process.py` 等
- **images：** 存储项目相关的图片，首页以及TIPC文档中需要的图像都可以放在这里，如果需要进一步区分功能，可以在里面建立不同的子文件夹。
- **model：** 存储模型相关代码文件的文件夹，保存模型的实现，如 `resnet.py、cyclegan.py` 等
- **utils：** 存储功能类相关文件的文件夹，如可视化，文件夹操作，模型保存与加载等
- **deploy：** 部署相关文件夹，目前包含PaddleInference推理文件夹以及PaddleServing服务部署文件夹
- **tools：** 工具类文件夹，包含训练、评估、预测、模型导出等代码文件
- **scripts：** 工具类文件夹，包含训练、评估、预测、模型导出等脚本文件
- **test_tipc：** 训练一体 (TIPC) 测试文件夹
- **README_en.md：** 中文版当前模型的使用说明，规范参考 README 内容要求
- **README.md：** 英文版当前模型的使用说明，规范参考 README 内容要求
- **LICENSE：** LICENSE文件

## 2. 功能实现

模型需要提供的功能包含：

- 训练：可以在GPU单机单卡、单机多卡、CPU多核的环境下执行训练
- 预测：可以在GPU单卡和CPU单核下执行预测
- 评估：可以在GPU单卡和CPU单核下执行评估
- 模型导出：可以导出inference模型，并且跑通PaddleInference推理以及PaddleServing部署
- 使用自定义数据：要求模型可以灵活支持/适配自定义数据，可以通过在README中加入数据格式描部分和如何使用自定义数据章节解决

### 3. 命名规范和使用规范

- 文件和文件夹命名中，尽量使用下划线`_`代表空格，不要使用`-`。
- 模型定义过程中，需要有一个统一的变量（parameter）命名管理手段，如尽量手动声明每个变量的名字并支持名称可变，禁止将名称定义为一个常数（如"embedding"），避免在复用代码阶段出现各种诡异的问题。
- 重要文件，变量的名称定义过程中需要能够通过名字表明含义，禁止使用含混不清的名称，如net.py, aaa.py等。
- 在代码中定义path时，需要使用os.path.join完成，禁止使用string加的方式，导致模型对windows环境缺乏支持。


### 4. 注释和License

对于代码中重要的部分，需要加入注释介绍功能，帮助用户快速熟悉代码结构，包括但不仅限于：

- Dataset、DataLoader的定义。
- 整个模型定义，包括input，运算过程，loss等内容。
- init，save，load，等io部分
- 运行中间的关键状态，如print loss，save model等。

如：
```
import random

from paddle.io import Dataset
from paddle.vision.transforms import transforms as T


class PetDataset(Dataset):
    """
    Pet 数据集定义
    """
    def __init__(self, mode='train'):
        """
        构造函数
        """
        self.image_size = IMAGE_SIZE
        self.mode = mode.lower()

        assert self.mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []

        with open('./{}.txt'.format(self.mode), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path, color_mode='rgb', transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = PilImage.open(io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

            return T.Compose([
                T.Resize(self.image_size)
            ] + transforms)(img)

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        train_image = self._load_img(self.train_images[idx],
                                     transforms=[
                                         T.Transpose(),
                                         T.Normalize(mean=127.5, std=127.5)
                                     ]) # 加载原始图像
        label_image = self._load_img(self.label_images[idx],
                                     color_mode='grayscale',
                                     transforms=[T.Grayscale()]) # 加载Label图像

        # 返回image, label
        train_image = np.array(train_image, dtype='float32')
        label_image = np.array(label_image, dtype='int64')
        return train_image, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images)
```


对于整个模型代码，都需要在文件头内加入licenses，readme中加入licenses标识。

文件头内licenses格式如下：

```
#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## 5. 其他问题

- 使用 Paddle 2.x API开发，不使用 `paddle.fluid.*` 下的API；
- 代码封装得当，易读性好，不用一些随意的变量/类/函数命名
- 注释清晰，不仅说明做了什么，也要说明为什么这么做
- 如果模型依赖paddlepaddle未涵盖的依赖（如 pandas），则需要在README中显示提示用户安装对应依赖
- 随机控制，需要尽量固定含有随机因素模块的随机种子，保证模型可以正常复现
- 超参数：模型内部超参数禁止写死，尽量都可以通过配置文件进行配置。

## 6. README 内容&格式说明

模型的readme共分为以下几个部分，**具体模板见：**[README.md](repo_template/README.md)

```
# 模型名称
## 1. 简介
## 2. 复现精度
## 3. 数据集
## 4. 环境依赖
## 5. 快速开始
## 6. 代码结构与详细说明
## 7. 模型信息
```
