# DuReader Dataset
DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

# DuReader Network
DuReader is inspired by 3 classic reading comprehension models([BiDAF](https://arxiv.org/abs/1611.01603), [Match-LSTM](https://arxiv.org/abs/1608.07905), [R_NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)).

Attention Flow from [BiDAF](https://arxiv.org/abs/1611.01603)

Anwser Point Network from [Match-LSTM](https://arxiv.org/abs/1608.07905)

Attention Pooling from [R_NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
## How to Run
### Download the Dataset
To Download DuReader dataset:
```
cd data && bash download.sh
```
For more details about DuReader dataset please refer to [DuReader Dataset Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader).

### Download Thirdparty Dependencies
We use Bleu and Rouge as evaluation metrics, the calculation of these metrics relies on the scoring scripts under "https://github.com/tylin/coco-caption", to download them, run:

```
cd utils && bash download_thirdparty.sh
```

### Preprocess the Data
After the dataset is downloaded, there is still some work to do to run the baseline systems. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our baseline models, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, you should first segment 'question', 'title', 'paragraphs' and then store the segemented result into 'segmented_question', 'segmented_title', 'segmented_paragraphs' like the downloaded preprocessed data, then run:
```
cat data/raw/trainset/search.train.json | python utils/preprocess.py > data/preprocessed/trainset/search.train.json
```
The preprocessed data can be automatically downloaded by `data/download.sh`, and is stored in `data/preprocessed`, the raw data before preprocessing is under `data/raw`.

### Run with PaddlePaddle

#### Get the Vocab File

Once the preprocessed data is ready, you can run `utils/get_vocab.py` to generate the vocabulary file, for example, if you want to train model with Baidu Search data:
```
python utils/get_vocab.py --files data/preprocessed/trainset/search.train.json data/preprocessed/devset/search.dev.json  --vocab data/vocab.search
```

If you want to use the demo data, run:
```
python utils/get_vocab.py --files data/demo/trainset/search.train.json data/demo/devset/search.dev.json  --vocab data/demo/vocab.search
```

#### Environment Requirements
For now we've only tested on PaddlePaddle v1.0, to install PaddlePaddle and for more details about PaddlePaddle, see [PaddlePaddle Homepage](http://paddlepaddle.org).

#### Training
The DuReader model can be trained by run `train.py`, for complete usage run `python train.py -h`.

The basic training and infering process has been wrapped in `run.sh`,  the basic usage is:
```
bash run.sh TASK_NAME
```
For example, to train the model, run:
```
bash run.sh train
```

## Run DuReader on multilingual datasets

To help evaluate the system performance on multilingual datasets, we provide scripts to convert MS MARCO V2 data from its format to DuReader format.

[MS MARCO](http://www.msmarco.org/dataset.aspx) (Microsoft Machine Reading Comprehension) is an English dataset focused on machine reading comprehension and question answering. The design of MS MARCO and DuReader is similiar. It is worthwhile examining the MRC systems on both Chinese (DuReader) and English (MS MARCO) datasets.

You can download MS MARCO V2 data, and run the following scripts to convert the data from MS MARCO V2 format to DuReader format. Then, you can run and evaluate our DuReader baselines or your DuReader systems on MS MARCO data.

```
./run_marco2dureader_preprocess.sh ../data/marco/train_v2.1.json ../data/marco/train_v2.1_dureaderformat.json
./run_marco2dureader_preprocess.sh ../data/marco/dev_v2.1.json ../data/marco/dev_v2.1_dureaderformat.json
```

## Copyright and License
Copyright 2017 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
