# Abstract
Dureader is an end-to-end neural networks model for machine reading comprehesion style question answering, which aims to anser questions from given passages. We first match the question and passage with a bidireactional attention flow networks to obtrain the question-aware passages represenation. Then we employ the pointer networks to locate the positions of answers from passages. Our experimental evalutions show that DuReader model achieves the state-of-the-art results in DuReader Dadaset.
# Dataset
DuReader Dataset is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

# Network
DuReader is inspired by 3 classic reading comprehension models([BiDAF](https://arxiv.org/abs/1611.01603), [Match-LSTM](https://arxiv.org/abs/1608.07905), [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)).

DuReader model is a hierarchical multi_stage process adn consist of five layers

- **Word Embedding Layer** maps each word to a vector space using a pre-trained word embedding model.
- **Encoding Layer** extract context infomation for each position in question and passages with bi-directional LSTM network.
- **Attention Flow Layer** couples the query and context vectors and produces a set of query-aware feature vectors for each word in the context. Please refer to [BiDAF](https://arxiv.org/abs/1611.01603) for more details.
- **Fusion Layer** employs two layers of bi-directional LSTM to capture the interaction among context words independent of the query.
- **Answer Point Network Layer with Attention Pooling** please refer to [Match-LSTM](https://arxiv.org/abs/1608.07905) and [R_NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) for more details.

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
After the dataset is downloaded, there is still some work to do to run DuReader. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our model, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, you should first segment 'question', 'title', 'paragraphs' and then store the segemented result into 'segmented_question', 'segmented_title', 'segmented_paragraphs' like the downloaded preprocessed data, then run:
```
cat data/raw/trainset/search.train.json | python utils/preprocess.py > data/preprocessed/trainset/search.train.json
```
The preprocessed data can be automatically downloaded by `data/download.sh`, and is stored in `data/preprocessed`, the raw data before preprocessing is under `data/raw`.

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
The DuReader model can be trained by run `run.py`, for complete usage run `python run.py -h`.

The basic training and infering process has been wrapped in `run.sh`,  the basic usage is:
```
bash run.sh --TASK_NAME
```
For example, to train the model, run:
```
bash run.sh --train
```
#### Inference
To infer a trained model, run the same command as training and change `train` to `infer`,  and add `--testset <path_to_testset>` argument. for example, suppose a model is successfully trained and parameters of the model are saved in a directory such as `models/1`, to infer the saved model, run:
```
bash run.sh --infer --testset ../data/preprocessed/testset/search.test.json --load_dir models/1  --result_dir infer
```
The result corresponding to the model saved is under `infer` folder, and the evaluation metrics is logged.

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
