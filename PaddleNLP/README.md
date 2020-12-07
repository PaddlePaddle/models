# PaddleNLP

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## Introduction

PaddleNLP aims to accelerate NLP applications by powerful model zoo, easy-to-use API and detailed tutorials, It's also the NLP best practice for PaddlePaddle 2.0 API system.

**TODO:** Add an architecture chart for PaddleNLP

## Features

* **Rich and Powerful Model Zoo**
  - Our Model Zoo covers mainstream NLP applications, including Lexical Analysis, Syntactic Parsing, Machine Translation, Text Classification, Text Generation, Text Matching, General Dialogue and Question Answering etc.
* **Easy-to-use API**
  - The API is fully integrated with PaddlePaddle high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, training and evaluation. which enables you to deal with text problems more productively.
* **High Performance and Large-scale Training**
  - We provide a highly optimized ditributed training implementation for BERT with Fleet API, it can fully utilize GPU clusters for large-scale model pre-training. Please refer to our [benchmark](./benchmark/bert) for more information.
* **Detailed Tutorials and Industrial Practices**
  - We offers detailed and interactable notebook tutorials to show you the best practices of PaddlePaddle 2.0.

## Installation

### Prerequisites

* python >= 3.6
* paddlepaddle >= 2.0.0-rc1

```
pip install paddlenlp
```

## Quick Start

### Quick Dataset Loading

```python
dataset = paddlenlp.datasets.ChnSentiCorp(split="train")
```

### Reusable Text Emebdding

```python
wordemb = paddlenlp.embedding.SkipGram("Text8")
wordemb("language")
>>> [1.0, 2.0, 3.0, ...., 5.0, 6.0]
```

### High Quality Chinsese Pre-trained Model

```python
from paddlenlp.transformer import ErnieModel
ernie = ErnieModel.from_pretrained("ernie-1.0-chinese")
sequence_output, pooled_output = ernie.forward(input_ids, segment_ids)
```

## Tutorials

List our notebook tutorials based on AI Studio.

## Community

* SIG for Pretrained Model Contribution
* SIG for Dataset Integration

## FAQ

## License

PaddleNLP is provided under the [Apache-2.0 license](./LICENSE).
