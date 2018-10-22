# Abstract
Dureader is an end-to-end neural network model for machine reading comprehension style question answering, which aims to answer questions from given passages. We first match the question and passages with a bidireactional attention flow network to obtrain the question-aware passages represenation. Then we employ a pointer network to locate the positions of answers from passages. Our experimental evalutions show that DuReader model achieves the state-of-the-art results in DuReader Dadaset.
# Dataset
DuReader Dataset is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

# Network
DuReader model is inspired by 3 classic reading comprehension models([BiDAF](https://arxiv.org/abs/1611.01603), [Match-LSTM](https://arxiv.org/abs/1608.07905), [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)).

DuReader model is a hierarchical multi-stage process and consists of five layers

- **Word Embedding Layer** maps each word to a vector using a pre-trained word embedding model.
- **Encoding Layer** extracts context infomation for each position in question and passages with a bi-directional LSTM network.
- **Attention Flow Layer** couples the query and context vectors and produces a set of query-aware feature vectors for each word in the context. Please refer to [BiDAF](https://arxiv.org/abs/1611.01603) for more details.
- **Fusion Layer** employs a layer of bi-directional LSTM to capture the interaction among context words independent of the query.
- **Decode Layer** employs an answer point network with attention pooling of the quesiton to locate the positions of answers from passages. Please refer to [Match-LSTM](https://arxiv.org/abs/1608.07905) and [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) for more details.

## How to Run
### Download the Dataset
To Download DuReader dataset:
```
cd data && bash download.sh
```
For more details about DuReader dataset please refer to [DuReader Dataset Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader).

### Download Thirdparty Dependencies
We use Bleu and Rouge as evaluation metrics, the calculation of these metrics relies on the scoring scripts under [coco-caption](https://github.com/tylin/coco-caption), to download them, run:

```
cd utils && bash download_thirdparty.sh
```
### Environment Requirements
For now we've only tested on PaddlePaddle v1.0, to install PaddlePaddle and for more details about PaddlePaddle, see [PaddlePaddle Homepage](http://paddlepaddle.org).

### Preparation
Before training the model, we have to make sure that the data is ready. For preparation, we will check the data files, make directories and extract a vocabulary for later use. You can run the following command to do this with a specified task name:

```
sh run.sh --prepare
```
You can specify the files for train/dev/test by setting the `trainset`/`devset`/`testset`.
### Training
To train the model and you can also set the hyper-parameters such as the learning rate by using `--learning_rate NUM`. For example, to train the model for 10 passes, you can run:

```
sh run.sh --train --pass_num 10
```

The training process includes an evaluation on the dev set after each training epoch. By default, the model with the least Bleu-4 score on the dev set will be saved.

### Evaluation
To conduct a single evaluation on the dev set with the the model already trained, you can run the following command:

```
sh run.sh --evaluate  --load_dir models/1
```

### Prediction
You can also predict answers for the samples in some files using the following command:

```
sh run.sh --predict --load_dir models/1 --testset ../data/preprocessed/testset/search.dev.json
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.
