# PaddleNLP

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## Introduction

PaddleNLP aims to accelerate NLP applications through powerful model zoo, easy-to-use API with detailed tutorials, It's also the NLP best practice for PaddlePaddle 2.0 API system.

** This project is still UNDER ACTIVE DEVELOPMENT. **

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
pip install paddlenlp==2.0.0a
```

## Quick Start

### Quick Dataset Loading

```python
train_ds, test_ds = paddlenlp.datasets.ChnSentiCorp.get_datasets(['train','test'])
```

### Reusable Text Emebdding

```python

from paddlenlp.embeddings import TokenEmbedding
wordemb = TokenEmbedding("word2vec.baike.300d")
print(wordemb.search("中国"))
>>> [0.260801  0.1047    0.129453 ... 0.096542  0.0092513]

```

### High Quality Chinsese Pre-trained Model

```python
from paddlenlp.transformers import ErnieModel
ernie = ErnieModel.from_pretrained("ernie")
sequence_output, pooled_output = ernie.forward(input_ids, segment_ids)
```

## Tutorials

List our notebook tutorials based on AI Studio.

## Community

* SIG for Pretrained Model Contribution
* SIG for Dataset Integration

## FAQ

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).