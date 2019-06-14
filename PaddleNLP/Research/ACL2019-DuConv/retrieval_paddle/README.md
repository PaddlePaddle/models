Knowledge-driven Dialogue
=============================
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a paddlepaddle implementation of retrieval-based model for knowledge-driven dialogue

## Requirements

* cuda=9.0
* cudnn=7.0
* python=2.7
* numpy
* paddlepaddle>=1.3

## Quickstart

### Step 1: Preprocess the data

Put the data of [DuConv](https://ai.baidu.com/broad/subordinate?dataset=duconv) under the data folder and rename them  train/dev/test.txt: 

```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

### Step 2: Train the model

Train model with the following commands. 

```bash
sh run_train.sh model_name
```

3 models were supported:
- match: match, input is history and response
- match_kn: match_kn, input is history, response, chat_path, knowledge
- match_kn_gene: match_kn, input is history, response, chat_path, knowledge and generalizes target_a/target_b of goal for all inputs, replaces them with slot mark

### Step 3: Test the Model

Test model with the following commands.

```bash
sh run_test.sh model_name
```

## Note !!!

* The script run_train.sh/run_test.sh shows all the processes including data processing and model training/testing. Be sure to read it carefully and follow it.
* The files in ./data and ./model is just empty file to show the structure of the document.
