# Deep Neural Networks for YouTube Recommendations

## Introduction\[[1](#References)\]
YouTube is the world's largest platform for creating, sharing and discovering video content. Youtube recommendations are responsible for helping more than a billion users discover personalized content from an ever-growing corpus of videos.
- Scale: Many existing recommendation algorithm proven to work well on small problems fail to operate on massive scale. Highly specialized distributed learning algorithms and efficient serving systems are essential.
- Freshness: YouTube has a very dynamic corpus with many hours of video are uploaded per second. The recommendation system should model newly uploaded content as well as the latest actions taken by user.
- Noise: Historical user behavior on YouTube is inherently difficult to predict due to sparsity and a variety of unobservable external factors. Furthermore, the noisy implicit feedback signals instead of the ground truth of user satisfaction is observed, and metadata associated with content is poorly structured, which forces the algorithms to be robust.

The overall structure of the recommendation system is illustrated in Figure 1.
<p align="center">
<img src="images/recommendation_system.png" width="500" height="300" hspace='10'/> <br/>
Figure 1. Recommendation system architecture
</p>

The system is comprised of two neural networks: one for candidate generation and one for ranking.
- The candidate generation network: It takes events from the user's YouTube activity history as input and retrieves a small subset(hundreds) of videos, highly relevant to the user, from a large corpus. The similarity between users is expressed in terms of coarse features such as IDs of video watches, search query tokens and demographics.
- The ranking network: It accomplishes this task by assigning a score to each video according to a desired objective function using a rich set of features describing the video and user.

## Candidate Generation
Here, candidate generation is modeled as extreme multiclass classification where the prediction problem becomes accurately classifying a specific video watch ![](https://www.zhihu.com/equation?tex=%5Comega_t) at time ![](https://www.zhihu.com/equation?tex=t) among millions of video ![](https://www.zhihu.com/equation?tex=i) (classes) from a corpus ![](https://www.zhihu.com/equation?tex=V) based on user ![](https://www.zhihu.com/equation?tex=U) and context ![](https://www.zhihu.com/equation?tex=C),

![](https://www.zhihu.com/equation?tex=%24P(%5Comega_t%3Di%7CU%2CC)%3D%5Cfrac%7Be%5E%7B%5Cmathbf%7Bv_i%7D%5Cmathbf%7Bu%7D%7D%7D%7B%5Csum_%7Bj%5Cin%20V%7D%5E%7B%20%7De%5E%7B%5Cmathbf%7Bv_j%7D%5Cmathbf%7Bu%7D%7D%7D)

where ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D%5Cin%20%5Cmathbb%7BR%7D%5EN) represents a high-dimensional "embedding" of the user, context pair and the ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv_j%7D%5Cin%20%5Cmathbb%7BR%7D%5EN) represent embeddings of each candidate video. The task of the deep neural network is to learn user embeddings ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D) as a function of the user's history and context that are useful for discriminating among videos with a softmax classifier.

Figure 2 shows the general network architecture of candidate generation model:
<p align="center">
<img src="images/model_network.png" width="600" height="500" hspace='10'/> <br/>
Figure 2. Candidate generation model architecture
</p>

- Input layer: A user's watch history is represented by a variable-length sequence of sparse video IDs, and search history is similarly represented by a variable-length sequence of search tokens.
- Embedding layer: The input features each is mapped to a fixed-sized dense vector representation via the embeddings, and then simply averaging the embeddings. The embeddings are learned jointly with all other model parameters through normal gradient descent back-propagation updates.
- Hidden layer: Features are concatenated into a wide first layer, followed by several layers of fully connected Rectified Linear Units (ReLU). The output of the last ReLU layer is the previous mentioned high-dimensional "embedding" of the user ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D), so called user vector.
- Output layer: A softmax classifier is connected to do discriminating millions of classes (videos). To speed up training process, a technique is applied that samples negative classes from background distribution with importance weighting. The previous mentioned high-dimensional "embedding" of the candidate video ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D) is obtained by weight and bias of the softmax layer. At serving time, the most likely N classes (videos) is computed for presenting to the user. To Score millions of items under a strict serving laterncy, the scoring problem reduces to a nearest neighbor search in the dot product space, and Locality Sensitive Hashing is relied on.

## Data Pre-processing
In this example, we moked click log of users as sample data, and its format is as follows:
```
user-id \t province \t city \t history-clicked-video-info-sequence \t phone

history-clicked-video-info-sequence is formated as
video-info1;video-info2;...;video-infoK

video-info is formated as
video-id:category:tag1_tag2_tag3_...tagM

For example:
USER_ID_15  Shanghai  Shanghai    VIDEO_42:CATEGORY_9:TAG115;VIDEO_43:CATEGORY_9:TAG116_TAG115;VIDEO_44:CATEGORY_2:TAG117_TAG71  GO T5
```
Run this code to prepare the sample data.
```
cd data
tar -zxvf data.tar
```

Then, run `data_preprocess.py` for data pre-processiong. Refer to the following instructions：
```
usage: data_processor.py [-h] --train_set_path TRAIN_SET_PATH --output_dir
                         OUTPUT_DIR [--feat_appear_limit FEAT_APPEAR_LIMIT]

PaddlePaddle Deep Candidate Generation Example

optional arguments:
  -h, --help            show this help message and exit
  --train_set_path TRAIN_SET_PATH
                        path of the train set
  --output_dir OUTPUT_DIR
                        directory to output
  --feat_appear_limit FEAT_APPEAR_LIMIT
                        the minimum number of feature values appears (default:
                        20)
```
The fucntion of this script is as follows:
- Filter low-frequency features\[[2](#References)\], which appears less than `feat_appear_limit` times.
- Encode features, and generate dictionary `feature_dict.pkl`.
- Count the probability of each video appears and write into `item_freq.pkl`, and provide it to NCE layer.

For example, run the following command to accomplish data pre-processing:
```
python data_processor.py --train_set_path=./data/train.txt \
                                     --output_dir=./output \
                                     --feat_appear_limit=20
```

## Model Implementaion
The details of model implementation is illustrated as follows. The code is in `./network_conf.py`.

### Input layer
```python
def _build_input_layer(self):
    """
    build input layer
    """
    self._history_clicked_items = paddle.layer.data(
        name="history_clicked_items", type=paddle.data_type.integer_value_sequence(
            len(self._feature_dict['history_clicked_items'])))
    self._history_clicked_categories = paddle.layer.data(
        name="history_clicked_categories", type=paddle.data_type.integer_value_sequence(
            len(self._feature_dict['history_clicked_categories'])))
    self._history_clicked_tags = paddle.layer.data(
        name="history_clicked_tags", type=paddle.data_type.integer_value_sequence(
            len(self._feature_dict['history_clicked_tags'])))
    self._user_id = paddle.layer.data(
        name="user_id", type=paddle.data_type.integer_value(
            len(self._feature_dict['user_id'])))
    self._province = paddle.layer.data(
        name="province", type=paddle.data_type.integer_value(
            len(self._feature_dict['province'])))
    self._city = paddle.layer.data(
        name="city", type=paddle.data_type.integer_value(len(self._feature_dict['city'])))
    self._phone = paddle.layer.data(
        name="phone", type=paddle.data_type.integer_value(len(self._feature_dict['phone'])))
    self._target_item = paddle.layer.data(
        name="target_item", type=paddle.data_type.integer_value(
            len(self._feature_dict['history_clicked_items'])))
```

### Embedding layer
The each of input features is mapped to a fixed-sized dense vector representation
```python
def _create_emb_attr(self, name):
    """
    create embedding parameter
    """
    return paddle.attr.Param(
        name=name, initial_std=0.001, learning_rate=1, l2_rate=0, sparse_update=True)

def _build_embedding_layer(self):
    """
    build embedding layer
    """
    self._user_id_emb = paddle.layer.embedding(input=self._user_id,
                                               size=64,
                                               param_attr=self._create_emb_attr(
                                                   '_proj_user_id'))
    self._province_emb = paddle.layer.embedding(input=self._province,
                                                size=8,
                                                param_attr=self._create_emb_attr(
                                                    '_proj_province'))
    self._city_emb = paddle.layer.embedding(input=self._city,
                                            size=16,
                                            param_attr=self._create_emb_attr('_proj_city'))
    self._phone_emb = paddle.layer.embedding(input=self._phone,
                                             size=16,
                                             param_attr=self._create_emb_attr('_proj_phone'))
    self._history_clicked_items_emb = paddle.layer.embedding(
        input=self._history_clicked_items,
        size=64,
        param_attr=self._create_emb_attr('_proj_history_clicked_items'))
    self._history_clicked_categories_emb = paddle.layer.embedding(
        input=self._history_clicked_categories,
        size=8,
        param_attr=self._create_emb_attr('_proj_history_clicked_categories'))
    self._history_clicked_tags_emb = paddle.layer.embedding(
        input=self._history_clicked_tags,
        size=64,
        param_attr=self._create_emb_attr('_proj_history_clicked_tags'))
```

### Hiddern layer
We improves the original networks in \[[1](#References)\] by modifying that the embeddings of video watches are not simply averaged but are connected to a LSTM layer with max temporal pooling instead, so that the deep sequential information related to user interests can be learned well. Considering data scale and efficiency of training, we only apply two ReLU layers, which also leads to good performance.

```python
self._rnn_cell = paddle.networks.simple_lstm(input=self._history_clicked_items_emb, size=64)
self._lstm_last = paddle.layer.pooling(
    input=self._rnn_cell, pooling_type=paddle.pooling.Max())
self._avg_emb_cats = paddle.layer.pooling(input=self._history_clicked_categories_emb,
                                          pooling_type=paddle.pooling.Avg())
self._avg_emb_tags = paddle.layer.pooling(input=self._history_clicked_tags_emb,
                                          pooling_type=paddle.pooling.Avg())
self._fc_0 = paddle.layer.fc(
    name="Relu1",
    input=[self._lstm_last, self._user_id_emb,
           self._city_emb, self._phone_emb],
    size=self._dnn_layer_dims[0],
    act=paddle.activation.Relu())

self._fc_1 = paddle.layer.fc(
    name="Relu2",
    input=self._fc_0,
    size=self._dnn_layer_dims[1],
    act=paddle.activation.Relu())
```

### Output layer
To speed up training process, Noise-contrastive estimation, NCE\[[3](#references)\] is applied to sample negative classes from background distribution with importance weighting. The previous mentioned `item_freq.pkl`[data pre-processing](#data pre-processing) is used as neg_distribution.
```python
return paddle.layer.nce(
                input=self._fc_1,
                label=self._target_item,
                num_classes=len(self._feature_dict['history_clicked_items']),
                param_attr=paddle.attr.Param(name="nce_w"),
                bias_attr=paddle.attr.Param(name="nce_b"),
                act=paddle.activation.Sigmoid(),
                num_neg_samples=5,
                neg_distribution=self._item_freq)
```

## Train
First of all, prepare `reader.py`, the function of which is to convert raw features into encoding id. One piece of train data generates several data instances according to `window_size`, and then is fed into trainer.
```
window_size=2
train data:
user-id \t province \t city \t video-info1;video-info2;...;video-infoK \t phone

several data instances:
user-id,province,city,[<unk>,video-id1],[<unk>,category1],[<unk>,tags1],phone,video-id2
user-id,province,city,[video-id1,video-id2],[category1,category2],[tags1,tags2],phone,video-id3
user-id,province,city,[video-id2,video-id3],[category2,category3],[tags2,tags3],phone,video-id4
......
```
The relevant code is as follows:
```python
for i in range(1, len(history_clicked_items_all)):
    start = max(0, i - self._window_size)
    history_clicked_items = history_clicked_items_all[start:i]
    history_clicked_categories = history_clicked_categories_all[start:i]
    history_clicked_tags_str = history_clicked_tags_all[start:i]
    history_clicked_tags = []
    for tags_a in history_clicked_tags_str:
        for tag in tags_a.split("_"):
            history_clicked_tags.append(int(tag))
    target_item = history_clicked_items_all[i]
    yield user_id, province, city, \
          history_clicked_items, history_clicked_categories, \
          history_clicked_tags, phone, target_item
```
```python
reader = Reader(feature_dict, args.window_size)
    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: reader.train(args.train_set_path),
                buf_size=7000), args.batch_size),
        num_passes=args.num_passes,
        feeding=feeding,
        event_handler=event_handler)
```
Then start training.
```shell
python train.py --train_set_path='./data/train.txt' \
    --test_set_path='./data/test.txt' \
    --model_output_dir='./output/model/' \
    --feature_dict='./output/feature_dict.pkl' \
    --item_freq='./output/item_freq.pkl'
```

## Offline prediction
Input user related features, and then get the most likely N videos for user.
```shell
python infer.py --infer_set_path='./data/infer.txt' \
    --model_path='./output/model/model_pass_00000.tar.gz' \
    --feature_dict='./output/feature_dict.pkl' \
    --batch_size=50
```

## Online prediction
For online prediction，we adopt Approximate Nearest Neighbor(ANN) to directly recall top N most likely watch video. However, our ANN system currently only supports cosin sorting, not by inner product sorting, which leads to big effect difference.

As a result, we sliently modify user and video vectors by a SIMPLE-LSH conversion\[[4](#References)\], so that inner sorting is equivalent to cosin sorting after conversion.

Details are as follows:
- For video vector ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D%5Cin%20%5Cmathbb%7BR%7D%5EN), we have ![](https://www.zhihu.com/equation?tex=%5Cleft%20%5C%7C%20%5Cmathbf%7Bv%7D%20%5Cright%20%5C%7C%5Cleqslant%20m). The modified video vector ![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bv%7D%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%2B1%7D), and let ![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bv%7D%7D%20%3D%20%5B%5Cfrac%7B%5Cmathbf%7Bv%7D%7D%7Bm%7D%3B%20%5Csqrt%7B1%20-%5Cleft%20%5C%7C%20%5Cmathbf%7B%5Cfrac%7B%5Cmathbf%7Bv%7D%7D%7Bm%7D%7B%7D%7D%20%5Cright%20%5C%7C%5E2%7D%5D).

- For user vector ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D%5Cin%20%5Cmathbb%7BR%7D%5EN), and the modified user vector ![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%2B1%7D), and let ![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%20%3D%20%5B%5Cmathbf%7Bu%7D_%7Bnorm%7D%3B%200%5D), where ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D_%7Bnorm%7D) is normalized ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D).

When online predicting, for a coming ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D), we need recall ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D) by inner product sorting. After ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D%5Crightarrow%20%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%2C%20%5Cmathbf%7Bv%7D%5Crightarrow%20%5Ctilde%7B%5Cmathbf%7Bv%7D%7D) conversion, the order of inner prodct sorting is unchanged. Since ![](https://www.zhihu.com/equation?tex=%5Cleft%20%5C%7C%20%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%20%5Cright%20%5C%7C) and ![](https://www.zhihu.com/equation?tex=%5Cleft%20%5C%7C%20%5Ctilde%7B%5Cmathbf%7Bv%7D%7D%20%5Cright%20%5C%7C) are both equal to 1, ![](https://www.zhihu.com/equation?tex=cos(%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%20%2C%5Ctilde%7B%5Cmathbf%7Bv%7D%7D)%20%3D%20%5Ctilde%7B%5Cmathbf%7Bu%7D%7D%5Ccdot%20%5Ctilde%7B%5Cmathbf%7Bv%7D%7D), which makes cosin-supported-only ANN system works.

And in order to retain precision, we find that ![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bv%7D%7D%3D%5B%5Cmathbf%7Bv%7D%3B%5Csqrt%7Bm%5E2-%5Cleft%5C%7C%20%5Cmathbf%7B%5Cmathbf%7Bv%7D%7D%5Cright%5C%7C%5E2%7D%5D) is also equivalent.

Use `user_vector.py` and `vector.py` to calculate user and item vectors. For example, run the following commands:
```shell
python user_vector.py --infer_set_path='./data/infer.txt' \
        --model_path='./output/model/model_pass_00000.tar.gz' \
            --feature_dict='./output/feature_dict.pkl' \
                --batch_size=50
python item_vector.py --model_path='./output/model/model_pass_00000.tar.gz' \
            --feature_dict='./output/feature_dict.pkl'
```

## References
1. Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
2. https://code.google.com/archive/p/word2vec/
3. http://paddlepaddle.org/docs/develop/models/nce_cost/README.html
4. Neyshabur, Behnam, and Nathan Srebro. "On symmetric and asymmetric LSHs for inner product search." arXiv preprint arXiv:1410.5518 (2014).
