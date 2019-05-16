PaddleNLP
=========

PaddleNLP 是百度开源的工业级 NLP 工具与预训练模型集，能够适应全面丰富的 NLP 任务，方便开发者灵活插拔尝试多种网络结构，并且让应用最快速达到工业级效果。

PaddleNLP 完全基于[PaddlePaddle Fluid](http://www.paddlepaddle.org/)开发，并提供依托于百度百亿级大数据的预训练模型，能够极大地方便 NLP 研究者和工程师快速应用。使用者可以用PaddleNLP 快速实现文本分类、文本匹配、序列标注、阅读理解、智能对话等NLP任务的组网、建模和部署，并且可以直接使用百度开源工业级预训练模型进行快速应用。用户在极大地减少研究和开发成本的同时，也可以获得更好的基于工业实践的应用效果。

特点与优势
--------
- 全面丰富的中文NLP应用任务；
- 任务与网络解耦，网络灵活可插拔；
- 强大的工业化预训练模型，打造优异应用效果。

目录结构
------

```text
.
├── dialogue_model_toolkit            # 对话模型工具箱
├── emotion_detection                 # 对话情绪识别
├── knowledge_driven_dialogue         # 知识驱动对话
├── language_model                    # 语言模型
├── language_representations_kit      # 语言表示工具箱
├── lexical_analysis                  # 词法分析
├── models                            # 共享网络
│   ├── __init__.py
│   ├── classification
│   ├── dialogue_model_toolkit
│   ├── language_model
│   ├── matching
│   ├── neural_machine_translation
│   ├── reading_comprehension
│   ├── representation
│   ├── sequence_labeling
│   └── transformer_encoder.py
├── neural_machine_translation        # 机器翻译
├── preprocess                        # 共享文本预处理工具
│   ├── __init__.py
│   ├── ernie
│   ├── padding.py
│   └── tokenizer
├── reading_comprehension             # 阅读理解
├── sentiment_classification          # 文本情感分析
├── similarity_net                    # 短文本语义匹配
```

其中，除了 `models` 和 `preprocess` 分别是共享的模型集合与共享的数据预处理流程之外，其它目录包含的都是相互独立的任务，可以直接进入这些目录运行任务。

快速安装
-------

### 依赖

本项目依赖于 Python 2.7 和 Paddle Fluid 1.3.1 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 安装 PaddlePaddle。

### 流程

- 克隆代码库到本地

```shell
git clone https://github.com/PaddlePaddle/models.git
```

- 进入到特定的子目录中查看代码和运行任务（如情感分析）

```shell
cd models/PaddleNLP/sentiment_classification 
```

支持的 NLP 任务
-------------

### 文本分类
 - [文本情感分析](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/sentiment_classification)
 - [对话情绪识别](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/emotion_detection)
 
### 文本匹配
 - [短文本语义匹配](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/similarity_net)

### 序列标注
 - [词法分析](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis)

### 文本生成
 - [机器翻译](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer)

### 语义表示与语言模型
 - [语言表示工具箱](https://github.com/PaddlePaddle/LARK/tree/develop)
 - [语言模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model)

### 复杂任务
 - [对话模型工具箱](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit)
 - [知识驱动对话](https://github.com/baidu/knowledge-driven-dialogue/tree/master)
 - [阅读理解](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/reading_comprehension)

